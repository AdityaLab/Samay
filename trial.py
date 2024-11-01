from model import TimesfmModel
from dataset import TimesfmDataset
import torch
import numpy as np


def main():
    dataset = TimesfmDataset(name="tycho", path='/nethome/sli999/data/Tycho/timesfm_US_covid_pivot.csv')

    repo = "google/timesfm-1.0-200m-pytorch"
    config = {
        "context_len": 128,
        "horizon_len": 32,
        "backend": "gpu",
        "per_core_batch_size": 32,
        "input_patch_len": 32,
        "output_patch_len": 128,
        "num_layers": 20,
        "model_dims": 1280,
        "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }
    
    tfm = TimesfmModel(config=config, repo=repo)
    finetuned_model = tfm.finetune(dataset)


if __name__ == "__main__":
    main()