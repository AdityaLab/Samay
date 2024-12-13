# Samay: Time-series Foundational Models Library

Package for training and evaluating time-series foundational models.

Current repository contains the following models:

1. [LPTM](https://arxiv.org/abs/2311.11413)
2. [MOMENT](https://arxiv.org/abs/2402.03885)
3. [TimesFM](https://arxiv.org/html/2310.10688v2)
4. [Chronos](https://arxiv.org/abs/2403.07815)
5. [MOIRAI](https://arxiv.org/abs/2402.02592)

More models will be added soon...

## Installation

You can add the package to your project by running the following command:

To use LPTM run the following command:

```bash
pip install git+https://github.com/AdityaLab/Samay.git@LPTM
```

To use all the other models run:
```bash
pip install git+https://github.com/AdityaLab/Samay.git
```



### Development workflow

To develop on the project, you can clone the repository and install the package in editable mode:

```bash

## Clone repo
git clone https://github.com/AdityaLab/Samay.git

## Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

## Install dependencies
uv sync
```



## Usage Example

### LPTM

_Note:_ Use the LPTM branch for best results `pip install git+https://github.com/AdityaLab/Samay.git@LPTM`.

#### Loading  Model

```python
from samay.model import LPTMModel
from samay.dataset import LPTMDataset

repo = "lptm"
config = {
    "context_len": 512,
    "horizon_len": 192,
    "backend": "gpu",
    "per_core_batch_size": 32,
    "domain": "electricity",
}

lptm = LPTMModel(config=config, repo=repo)
```

#### Loading Dataset

```python
train_dataset = LPTMDataset(name="electricity", datetime_col='date', path='data/ETTh1.csv', 
                              mode='train', context_len=config["context_len"], horizon_len=128)
val_dataset = LPTMDataset(name="electricity", datetime_col='date', path='data/ETTh1.csv',
                                mode='test', context_len=config["context_len"], horizon_len=config["horizon_len"])
```

#### Zero-Forecasting

```python
avg_loss, trues, preds, histories = lptm.evaluate(val_dataset)
```

### TimesFM

Install the package: `pip install git+https://github.com/AdityaLab/Samay.git`.

#### Loading  Model

```python
from samay.model import TimesfmModel
from samay.dataset import TimesfmDataset

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

#### Loading Dataset

```python
train_dataset = TimesfmDataset(name="ett", datetime_col='date', path='data/ETTh1.csv', 
                              mode='train', context_len=config["context_len"], horizon_len=128)
val_dataset = TimesfmDataset(name="ett", datetime_col='date', path='data/ETTh1.csv',
                              mode='test', context_len=config["context_len"], horizon_len=config["horizon_len"])
```

#### Zero-Forecasting

```python
avg_loss, trues, preds, histories = tfm.evaluate(val_dataset)
```

### Support

Tested on Python 3.12, 3.13 on Linux (CPU + GPU) and MacOS (CPU). Supports NVIDIA GPUs.
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

## Contact

If you have any feedback or questions, you can contact us via email: hkamarthi3@gatech.edu.
