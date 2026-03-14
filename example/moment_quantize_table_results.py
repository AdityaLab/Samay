import os
import sys
import time
import copy
import torch
import numpy as np
from sklearn.metrics import mean_squared_error



src_path = os.path.abspath(os.path.join("..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from samay.model import MomentModel
from samay.dataset import MomentDataset

import torch
import torch.nn as nn

torch.backends.quantized.engine = "qnnpack"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quantize_linear_layers(model, quantization_type="int8"):

    if quantization_type == "int8":
        return torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )

    elif quantization_type == "float16":
        return torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.float16
        )

    else:
        raise ValueError("Unsupported quantization type")
    
def quantize(moment_model, quant_type="int8", device="cpu"):
    
    moment_model.model.eval()
    moment_model.model = moment_model.model.to(device)

    with torch.no_grad():
        moment_model.model = quantize_linear_layers(
            moment_model.model,
            quantization_type=quant_type
        )

    return moment_model.model


repo = "AutonLab/MOMENT-1-large"

config = {
    "task_name": "forecasting",
    "forecast_horizon": 192,
    "head_dropout": 0.1,
    "weight_decay": 0,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,
}

base_model = MomentModel(config=config, repo=repo)

val_dataset = MomentDataset(
    name="ett",
    datetime_col="date",
    path="./src/samay/models/moment/data/ETTh1.csv",
    mode="test",
    horizon_len=192,
    freq=None,
)

# Create models

fp32_model = copy.deepcopy(base_model)

fp16_model = copy.deepcopy(base_model)
fp16_model.model = fp16_model.model.half().to(device)

int8_model = copy.deepcopy(base_model)
quantize(int8_model, "int8", device="cpu")

print("THIS IS DONE YES!!!!")


# Evaluation functions


def compute_mse(moment_model, dataset):

    model = moment_model.model
    model.eval()

    run_device = device
    model.to(run_device)

    preds = []
    trues = []

    with torch.no_grad():
        for i in range(len(dataset)):

            sample = dataset[i]

            x = sample[0]          # (64, 512)
            y_future = sample[2]   # (64, 192)

            dtype = next(model.parameters()).dtype
            x = torch.tensor(x, dtype=dtype).unsqueeze(0).to(run_device)

            output = model(x_enc=x)
            pred = output.forecast.squeeze()  # (64,192)

            preds.append(pred.cpu().numpy().reshape(-1))
            trues.append(np.array(y_future).reshape(-1))

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    print("Final prediction shape:", preds.shape)
    print("Final target shape:", trues.shape)

    return mean_squared_error(trues, preds)

def model_size(moment_model):

    torch.save(moment_model.model.state_dict(), "temp.pt")
    size = os.path.getsize("temp.pt") / 1e6
    os.remove("temp.pt")

    return size


def inference_time(moment_model, dataset, runs=10):

    model = moment_model.model
    model.eval()

    run_device = device
    model.to(run_device)

    sample = dataset[0]
    x = sample[0]

    dtype = next(model.parameters()).dtype
    x = torch.tensor(x, dtype=dtype).unsqueeze(0).to(run_device)

    start = time.time()

    for _ in range(runs):
        with torch.no_grad():
            model(x_enc=x).forecast

    end = time.time()

    return (end - start) / runs


# Run experiments

print("Evaluating FP32...")
mse_fp32 = compute_mse(fp32_model, val_dataset)
size_fp32 = model_size(fp32_model)
time_fp32 = inference_time(fp32_model, val_dataset)

print("Evaluating FP16...")
mse_fp16 = compute_mse(fp16_model, val_dataset)
size_fp16 = model_size(fp16_model)
time_fp16 = inference_time(fp16_model, val_dataset)

print("Evaluating INT8...")
mse_int8 = compute_mse(int8_model, val_dataset)
size_int8 = model_size(int8_model)
time_int8 = inference_time(int8_model, val_dataset)


speedup_fp16 = time_fp32 / time_fp16
speedup_int8 = time_fp32 / time_int8

# RESULTS

print("\nMOMENT Results (ETTh1, Horizon=192)")
print("-------------------------------------")

print(f"Float32  | MSE: {mse_fp32:.5f} | Size: {size_fp32:.2f} MB | Speedup: 1.0x")
print(f"Float16  | MSE: {mse_fp16:.5f} | Size: {size_fp16:.2f} MB | Speedup: {speedup_fp16:.2f}x")
print(f"INT8     | MSE: {mse_int8:.5f} | Size: {size_int8:.2f} MB | Speedup: {speedup_int8:.2f}x")