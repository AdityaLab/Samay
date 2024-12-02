# Samay: Time-series Foundational Models Library

Package for training and evaluating time-series foundational models.

Current repository contains the following models:

1. [TimesFM](https://arxiv.org/html/2310.10688v2)
2. [MOMENT](https://arxiv.org/abs/2402.03885)
3. [LPTM](https://openreview.net/forum?id=vMMzjCr5Zj)
4. [Chronos](https://arxiv.org/abs/2403.07815)

More models will be added soon...

## Installation

You can add the package to your project by running the following command:

```bash
pip install git+https://github.com/AdityaLab/Samay.git
```

### Development workflow

To develop on the project, you can clone the repository and install the package in editable mode:

```bash

## Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

## Install dependencies
uv sync
```



## Usage Example

### Loading  Model

```python
from tsfmproject.model import TimesfmModel
from tsfmproject.dataset import TimesfmDataset

repo = "google/timesfm-1.0-200m-pytorch"
config = {
    "context_len": 512,
    "horizon_len": 192,
    "backend": "gpu",
    "per_core_batch_size": 32,
    "input_patch_len": 32,
    "output_patch_len": 128,
    "num_layers": 20,
    "model_dims": 1280,
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}

tfm = TimesfmModel(config=config, repo=repo)
```

### Loading Dataset

```python
train_dataset = TimesfmDataset(name="ett", datetime_col='date', path='data/ETTh1.csv', 
                              mode='train', context_len=config["context_len"], horizon_len=128)
val_dataset = TimesfmDataset(name="ett", datetime_col='date', path='data/ETTh1.csv',
                              mode='test', context_len=config["context_len"], horizon_len=config["horizon_len"])
```

### Zero-Forecasting

```python
avg_loss, trues, preds, histories = tfm.evaluate(val_dataset)
```

### Support

Tested on Python 3.12, 3.13 on Linux and MacOS. Supports NVIDIA GPUs.
Support for Windows and Apple Silicon GPUs is planned.

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@inproceedings{
kamarthi2024large,
title={Large Pre-trained time series models for cross-domain Time series analysis tasks},
author={Harshavardhan Kamarthi and B. Aditya Prakash},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=vMMzjCr5Zj}
}
```

