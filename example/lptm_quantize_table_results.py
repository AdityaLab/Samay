import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


src_path = os.path.abspath(os.path.join("..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from samay.dataset import LPTMDataset
from samay.model import LPTMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# torch.backends.quantized.engine = "qnnpack"


# Quantization


def quantize_linear_layers(model, quantization_type="int8"):

    if quantization_type == "int8":
        return torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

    elif quantization_type == "float16":
        return torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.float16
        )

    else:
        raise ValueError("Unsupported quantization type")


def quantize(lptm_model, quant_type="int8"):

    lptm_model.model.eval()
    lptm_model.model = lptm_model.model.to("cpu")

    with torch.no_grad():
        lptm_model.model = quantize_linear_layers(
            lptm_model.model, quantization_type=quant_type
        )

    return lptm_model.model


def quantize_linear_layers_bnb(module, threshold=6.0, quantization_type="int8"):

    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and child.in_features >= 128:
            if quantization_type == "int8":
                quant_layer = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=(child.bias is not None),
                    threshold=threshold,
                    has_fp16_weights=False,
                )
            elif quantization_type == "nf4":
                quant_layer = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=(child.bias is not None),
                    quant_type="nf4",
                    compute_dtype=torch.float16,
                )
            else:
                raise ValueError("Unsupported CUDA quantization type")

            with torch.no_grad():
                quant_layer.weight.copy_(child.weight)
                if child.bias is not None:
                    quant_layer.bias.copy_(child.bias)

            setattr(module, name, quant_layer)
        else:
            quantize_linear_layers_bnb(
                child, threshold=threshold, quantization_type=quantization_type
            )

    return module


def quantize_cuda(lptm_model, quant_type="int8"):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for CUDA quantization")
    if bnb is None:
        raise RuntimeError("bitsandbytes is not installed")

    lptm_model.model.eval()
    lptm_model.model = lptm_model.model.to("cuda")

    with torch.no_grad():
        lptm_model.model = quantize_linear_layers_bnb(
            lptm_model.model, quantization_type=quant_type
        )
        lptm_model.model = lptm_model.model.to("cuda")

    return lptm_model.model


def _resolve_run_device(lptm_model):

    return getattr(lptm_model, "_run_device", device)


def _resolve_input_dtype(lptm_model):

    model = lptm_model.model
    forced_dtype = getattr(lptm_model, "_input_dtype", None)
    if forced_dtype is not None:
        return forced_dtype

    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


# Load LPTM

config = {
    "task_name": "forecasting",
    "forecast_horizon": 192,
    "head_dropout": 0,
    "weight_decay": 0,
    "max_patch": 16,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
    "freeze_segment": True,
}

base_model = LPTMModel(config)


train_dataset = LPTMDataset(
    name="ett",
    datetime_col="date",
    path="./data/data/ETTh1.csv",
    mode="train",
    horizon=192,
)

val_dataset = LPTMDataset(
    name="ett",
    datetime_col="date",
    path="./data/data/ETTh1.csv",
    mode="test",
    horizon=192,
)

# Finetune

# print("Finetuning LPTM...")
# base_model = base_model.finetune(train_dataset)


# Create models

fp32_model = copy.deepcopy(base_model)
fp32_model._run_device = device
fp32_model._input_dtype = torch.float32

fp16_model = copy.deepcopy(base_model)
fp16_model.model = fp16_model.model.half().to(device)
fp16_model._run_device = device
fp16_model._input_dtype = torch.float16

int8_model = copy.deepcopy(base_model)
if torch.cuda.is_available() and bnb is not None:
    quantize_cuda(int8_model, "int8")
    int8_model._run_device = torch.device("cuda")
    int8_model._input_dtype = torch.float32
    print("Using bitsandbytes INT8 on CUDA")
else:
    quantize(int8_model, "int8")
    int8_model._run_device = torch.device("cpu")
    int8_model._input_dtype = torch.float32
    print("Falling back to torch dynamic INT8 on CPU")

print("Model setup complete.")


# Evaluation Functions


def compute_mse(lptm_model, dataset):

    model = lptm_model.model
    model.eval()

    run_device = _resolve_run_device(lptm_model)
    model.to(run_device)

    preds = []
    trues = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]

            x = sample[0]
            y_future = sample[2]

            dtype = _resolve_input_dtype(lptm_model)
            x = torch.tensor(x, dtype=dtype).unsqueeze(0).to(run_device)

            output = model(x_enc=x)
            pred = output.forecast.squeeze()

            preds.append(pred.cpu().numpy().reshape(-1))
            trues.append(np.array(y_future).reshape(-1))

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    print("Prediction shape:", preds.shape)
    print("Target shape:", trues.shape)

    return mean_squared_error(trues, preds)


def model_size(lptm_model):

    torch.save(lptm_model.model.state_dict(), "temp.pt")
    size = os.path.getsize("temp.pt") / 1e6
    os.remove("temp.pt")

    return size


def inference_time(lptm_model, dataset, runs=10):

    model = lptm_model.model
    model.eval()

    run_device = _resolve_run_device(lptm_model)
    model.to(run_device)

    sample = dataset[0]
    x = sample[0]

    dtype = _resolve_input_dtype(lptm_model)
    x = torch.tensor(x, dtype=dtype).unsqueeze(0).to(run_device)

    for _ in range(3):
        with torch.no_grad():
            model(x_enc=x).forecast

    start = time.time()

    for _ in range(runs):
        with torch.no_grad():
            model(x_enc=x).forecast

    end = time.time()

    return (end - start) / runs


# Run Experiments

print("\nEvaluating FP32...")
mse_fp32 = compute_mse(fp32_model, val_dataset)
size_fp32 = model_size(fp32_model)
time_fp32 = inference_time(fp32_model, val_dataset)

print("\nEvaluating FP16...")
mse_fp16 = compute_mse(fp16_model, val_dataset)
size_fp16 = model_size(fp16_model)
time_fp16 = inference_time(fp16_model, val_dataset)

print("\nEvaluating INT8...")
mse_int8 = compute_mse(int8_model, val_dataset)
size_int8 = model_size(int8_model)
time_int8 = inference_time(int8_model, val_dataset)


# Results

speedup_fp16 = time_fp32 / time_fp16
speedup_int8 = time_fp32 / time_int8

print("\nLPTM Results (ETTh1, Horizon=192)")
print("-------------------------------------")

print(f"Float32  | MSE: {mse_fp32:.5f} | Size: {size_fp32:.2f} MB | Speedup: 1.0x")
print(
    f"Float16  | MSE: {mse_fp16:.5f} | Size: {size_fp16:.2f} MB | Speedup: {speedup_fp16:.2f}x"
)
print(
    f"INT8     | MSE: {mse_int8:.5f} | Size: {size_int8:.2f} MB | Speedup: {speedup_int8:.2f}x"
)
