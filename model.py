import timesfm.src.timesfm as tfm
from timesfm.src.timesfm import pytorch_patched_decoder as ppd
import numpy as np
import pandas as pd
import torch


class Basemodel():
    def __init__(self, config=None, repo=None):
        """
        Args:
            config: dict, model configuration
            repo: str, Huggingface model repository id
        """
        self.config = config

    def finetune(self, dataloader, **kwargs):
        raise NotImplementedError

    def forecast(self, input, **kwargs):
        raise NotImplementedError

    def evaluate(self, X, y):
        pass

    def save(self, path):
        pass


class TimesfmModel(Basemodel):
    def __init__(self, config=None, repo=None, hparams=None, ckpt=None, **kwargs):
        super().__init__(config=config, repo=repo)
        hparams = tfm.TimesFmHparams(**self.config)
        if repo:
            try:
                ckpt = tfm.TimesFmCheckpoint(huggingface_repo_id=repo)
            except:
                raise ValueError(f"Repository {repo} not found")

        self.model = tfm.TimesFm(hparams=hparams, checkpoint=ckpt)

    def finetune(self, dataloader, **kwargs):
        """
        Args:
            dataloader: torch.utils.data.DataLoader, input data
        Returns:    
            FinetuneModel: ppd.PatchedDecoderFinetuneModel, finetuned model
        """
        core_layer_tpl = self.model._model
        # Todo: whether add freq
        FinetuneModel = ppd.PatchedDecoderFinetuneModel(core_layer_tpl=core_layer_tpl)
        FinetuneModel.train()
        optimizer = torch.optim.Adam(FinetuneModel.parameters(), lr=1e-3)
        for i, (inputs) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = FinetuneModel.compute_predictions(inputs)
            loss = FinetuneModel.compute_loss(outputs)
            loss.backward()
            optimizer.step()
        return FinetuneModel

    def forecast(self, input, **kwargs):
        """
        Args:
            input: torch.Tensor, input data
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - the mean forecast of size (# inputs, # forecast horizon),
                - the full forecast (mean + quantiles) of size
                (# inputs,  # forecast horizon, 1 + # quantiles).
        """
        return self.model.forecast(input)
    

class ChronosModel(Basemodel):
    def __init__(self, config=None, repo=None, hparams=None, ckpt=None, **kwargs):
        super().__init__(name="chronos", config=config, repo=repo)
        # Todo: load model

    def finetune(self, dataloader, **kwargs):
        # Todo: finetune model
        pass

    def forecast(self, input, **kwargs):
        # Todo: forecast
        pass
    


if __name__ == "__main__":
    name = "timesfm"
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
    print(tfm.model)


