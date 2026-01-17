import os
import torch
import torch.nn as nn
import bitsandbytes as bnb

#CHANGE THESE
USE_CUDA = False
QUANT_TYPE = "int8"

DEVICE = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
print("Using device:", DEVICE)
print("Quantization type:", QUANT_TYPE)


def quantize_linear_layers(module, threshold=6.0, quantization_type="int8"):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and child.in_features >= 128:
            if quantization_type == "int8":
                q = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=(child.bias is not None),
                    threshold=threshold,
                    has_fp16_weights=False,
                )
            elif quantization_type == "nf4":
                q = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=(child.bias is not None),
                    quant_type="nf4",
                    compute_dtype=torch.float16,
                )
            with torch.no_grad():
                q.weight.copy_(child.weight)
                if child.bias is not None:
                    q.bias.copy_(child.bias)
            setattr(module, name, q)
        else:
            quantize_linear_layers(child, threshold=threshold, quantization_type=quantization_type)
    return module


#LOADING MODEL -- FROM THE EXAMPLE LPTM NOTEBOOk
from samay.model import LPTMModel

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

lptm = LPTMModel(config)
lptm.model = lptm.model.to(DEVICE)


#QUANTIZATION
print("Before quantization (bytes):",
      sum(p.numel() * p.element_size() for p in lptm.model.parameters()))

lptm.model = quantize_linear_layers(lptm.model)
lptm.model = lptm.model.to(DEVICE)


#CHECKING IF QUANTIZATION IS SUCCESSFUL
def proof_report_lptm(lptm):
    print("PROOF REPORT: bitsandbytes quantization (LPTM)")

    m = lptm.model

    n8 = sum(1 for x in m.modules() if isinstance(x, bnb.nn.Linear8bitLt))
    n4 = sum(1 for x in m.modules() if isinstance(x, bnb.nn.Linear4bit))
    nlin = sum(1 for x in m.modules()
               if isinstance(x, nn.Linear)
               and not isinstance(x, bnb.nn.Linear8bitLt))

    print("Linear8bitLt layers:", n8)
    print("Linear4bit layers:  ", n4)
    print("Pure nn.Linear left:", nlin)

    if QUANT_TYPE == "int8" and n8 == 0:
        raise RuntimeError("INT8 quantization FAILED")

    if QUANT_TYPE == "nf4" and n4 == 0:
        raise RuntimeError("NF4 quantization FAILED")

    for name, layer in m.named_modules():
        if isinstance(layer, (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)):
            with torch.no_grad():
                x = torch.randn(4, layer.in_features, device=DEVICE)
                y = layer(x)
            print("Tested layer:", name, "->", tuple(y.shape))
            break

    print("PASS: Quantized layers exist and execute\n")


proof_report_lptm(lptm)


#EVALUATION WITH MEMORY TRACKING
from samay.dataset import LPTMDataset

val_dataset = LPTMDataset(
    name="ett",
    datetime_col="date",
    path="data/data/ETTh1.csv",
    mode="train",
    horizon=192,
)


def gpu_mem_mb():
    if DEVICE.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / 1024**2


if DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

print("GPU memory before eval (MB):", gpu_mem_mb())

try:
    print("Evaluating model...")
    avg_loss, trues, preds, histories = lptm.evaluate(
        val_dataset,
        task_name="forecasting"
    )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    print("GPU peak memory during eval (MB):", gpu_mem_mb())
    print("Inference SUCCESS")
    print("Avg loss:", avg_loss)
    print("Num predictions:", len(preds) if hasattr(preds, "__len__") else "n/a")

except Exception as e:
    print("Inference FAILED")
    print("Error:", repr(e))
