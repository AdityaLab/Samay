import timesfm.src.timesfm as tfm
from timesfm.src.timesfm import pytorch_patched_decoder as ppd
from moment.momentfm.models.moment import MOMENT, MOMENTPipeline
import numpy as np
import pandas as pd
import torch
from utils import get_least_used_gpu


class Basemodel():
    def __init__(self, config=None, repo=None):
        """
        Args:
            config: dict, model configuration
            repo: str, Huggingface model repository id
        """
        self.config = config
        least_used_gpu = get_least_used_gpu()
        if least_used_gpu >= 0:
            self.device = torch.device(f"cuda:{least_used_gpu}")
        else:
            self.device = torch.device("cpu")

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
    

    def finetune(self, dataset, **kwargs):
        """
        Args:
            dataloader: torch.utils.data.DataLoader, input data
        Returns:    
            FinetuneModel: ppd.PatchedDecoderFinetuneModel, finetuned model
        """
        core_layer_tpl = self.model._model
        # Todo: whether add freq
        FinetuneModel = ppd.PatchedDecoderFinetuneModel(core_layer_tpl=core_layer_tpl)
        FinetuneModel.to(self.device)
        FinetuneModel.train()
        dataloader = dataset.get_data_loader()
        optimizer = torch.optim.Adam(FinetuneModel.parameters(), lr=1e-4)
        epoch = 10 if 'epoch' not in kwargs else kwargs['epoch']
        avg_loss = 0
        for epoch in range(epoch):
            for i, (inputs) in enumerate(dataloader):
                inputs = dataset.preprocess_train_batch(inputs)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                optimizer.zero_grad()
                outputs = FinetuneModel.compute_predictions(inputs)
                loss = FinetuneModel.compute_loss(outputs, inputs)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            avg_loss /= len(dataloader)
            print(f"Epoch {epoch}, Loss: {avg_loss}")
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
        super().__init__(config=config, repo=repo)
        # Todo: load model

    def finetune(self, dataloader, **kwargs):
        # Todo: finetune model
        pass

    def forecast(self, input, **kwargs):
        # Todo: forecast
        pass


class MomentModel(Basemodel):
    def __init__(self, config=None, repo=None):
        super().__init__(config=config, repo=repo)
        if not repo:
            raise ValueError("Moment model requires a repository")
        self.model = MOMENTPipeline.from_pretrained(
            repo, 
            model_kwargs=self.config
        )
        self.model.init()

    def finetune(self, dataset, **kwargs):
        # arguments
        max_lr = 1e-4 if 'max_lr' not in kwargs else kwargs['max_lr']
        max_epoch = 2 if 'max_epoch' not in kwargs else kwargs['max_epoch']
        max_norm = 5.0 if 'max_norm' not in kwargs else kwargs['max_norm']

        dataloader = dataset.get_data_loader()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=max_lr)
        criterion.to(self.device)
        scaler = torch.amp.GradScaler()

        total_steps = len(dataloader) * max_epoch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3)
        self.model.to(self.device)
        self.model.train()

        for epoch in range(max_epoch):
            losses = []
            for i, data in enumerate(dataloader):
                # unpack the data
                timeseries, forecast, input_mask = data
                # Move the data to the GPU
                timeseries = timeseries.float().to(self.device)
                input_mask = input_mask.to(self.device)
                forecast = forecast.float().to(self.device)

                with torch.amp.autocast(device_type='cuda'):
                    output = self.model(x_enc=timeseries, input_mask=input_mask)
                
                loss = criterion(output.forecast, forecast)

                # Scales the loss for mixed precision training
                scaler.scale(loss).backward()

                # Clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                losses.append(loss.item())

            losses = np.array(losses)
            average_loss = np.average(losses)
            print(f"Epoch {epoch}: Train loss: {average_loss:.3f}")

            scheduler.step()

        return self.model




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
    model = tfm.model
    # print(tfm.model)
    df = pd.read_csv("/nethome/sli999/data/Tycho/dengue_laos.csv")
    df = df[df['SourceName'] == 'Laos Dengue Surveillance System']
    df = df[['Admin1ISO', 'PeriodStartDate', 'CountValue']]
    df.columns = ['unique_id', 'ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values(by=['unique_id', 'ds'])
    forecast_df = model.forecast_on_df(
        inputs=df,
        freq="D",  # daily frequency
        value_name="y",
        num_jobs=1,
    )
    forecast_df = forecast_df[['ds', 'unique_id', 'timesfm']]
    forecast_df.columns = ['ds', 'unique_id', 'y']

    print(forecast_df.head())

