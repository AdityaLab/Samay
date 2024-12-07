from .models.timesfm import timesfm as tfm
from .models.timesfm.timesfm import pytorch_patched_decoder as ppd
from .models.moment.momentfm.models.moment import MOMENTPipeline
import numpy as np
import pandas as pd
import torch

from .utils import get_least_used_gpu
from .models.moment.momentfm.utils.masking import Masking


class Basemodel:
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

    def finetune(self, dataset, **kwargs):
        raise NotImplementedError

    def forecast(self, input, **kwargs):
        raise NotImplementedError

    def evaluate(self, dateset, **kwargs):
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

    def finetune(self, dataset, freeze_transformer=True, **kwargs):
        """
        Args:
            dataloader: torch.utils.data.DataLoader, input data
        Returns:
            FinetuneModel: ppd.PatchedDecoderFinetuneModel, finetuned model
        """
        core_layer_tpl = self.model._model
        # Todo: whether add freq
        FinetunedModel = ppd.PatchedDecoderFinetuneModel(core_layer_tpl=core_layer_tpl)
        if freeze_transformer:
            for param in FinetunedModel.core_layer.stacked_transformer.parameters():
                param.requires_grad = False
        FinetunedModel.to(self.device)
        FinetunedModel.train()
        dataloader = dataset.get_data_loader()
        optimizer = torch.optim.Adam(FinetunedModel.parameters(), lr=1e-4)
        epoch = 10 if "epoch" not in kwargs else kwargs["epoch"]
        avg_loss = 0
        for epoch in range(epoch):
            for i, (inputs) in enumerate(dataloader):
                inputs = dataset.preprocess(inputs)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                optimizer.zero_grad()
                outputs = FinetunedModel.compute_predictions(inputs) # b, n, seq_len, 1+quantiles
                loss = FinetunedModel.compute_loss(outputs, inputs)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            avg_loss /= len(dataloader)
            print(f"Epoch {epoch}, Loss: {avg_loss}")
        
        self.model._model = FinetunedModel.core_layer
        return self.model

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
    
    def evaluate(self, dataset, **kwargs):
        dataloader = dataset.get_data_loader()
        trues, preds, histories, losses = [], [], [], []
        with torch.no_grad():
            for i, (inputs) in enumerate(dataloader):
                inputs = dataset.preprocess(inputs)
                input_ts = inputs["input_ts"]
                input_ts = np.squeeze(input_ts)
                actual_ts = inputs["actual_ts"].detach().cpu().numpy()
                actual_ts = np.squeeze(actual_ts)

                output, _ = self.model.forecast(input_ts)
                output = output[:, 0:actual_ts.shape[1]]

                loss = np.mean((output - actual_ts) ** 2)
                losses.append(loss.item())
                trues.append(actual_ts)
                preds.append(output)
                histories.append(input_ts)

        losses = np.array(losses)
        average_loss = np.average(losses)
        trues = np.stack(trues, axis=0)
        preds = np.stack(preds, axis=0)
        histories = np.stack(histories, axis=0)

        return average_loss, trues, preds, histories

                
        
        


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

    def finetune(self, dataset, task_name="forecasting", **kwargs):
        # arguments
        max_lr = 1e-4 if 'lr' not in kwargs else kwargs['lr']
        max_epoch = 2 if 'epoch' not in kwargs else kwargs['epoch']
        max_norm = 5.0 if 'norm' not in kwargs else kwargs['norm']
        mask_ratio = 0.25 if 'mask_ratio' not in kwargs else kwargs['mask_ratio']

        if task_name == "imputation" or task_name == "detection":
            mask_generator = Masking(mask_ratio=mask_ratio)

        dataloader = dataset.get_data_loader()
        criterion = torch.nn.MSELoss()
        if task_name == "classification":
            criterion = torch.nn.CrossEntropyLoss()
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
                if task_name == "forecasting":
                    timeseries, input_mask, forecast = data
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(self.device)
                    input_mask = input_mask.to(self.device)
                    forecast = forecast.float().to(self.device)
                    with torch.amp.autocast(device_type='cuda'):
                        output = self.model(x_enc=timeseries, input_mask=input_mask)
                    loss = criterion(output.forecast, forecast)

                elif task_name == "imputation":
                    timeseries, input_mask = data
                    n_channels = timeseries.shape[1]
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(self.device)
                    timeseries = timeseries.reshape(-1, 1, timeseries.shape[-1])
                    input_mask = input_mask.to(self.device).long()
                    input_mask = input_mask.repeat_interleave(n_channels, axis=0)
                    mask = mask_generator.generate_mask(x=timeseries, input_mask=input_mask).to(self.device).long()
                    output = self.model(x_enc=timeseries, input_mask=input_mask, mask=mask)
                    with torch.amp.autocast(device_type='cuda'):
                        recon_loss = criterion(output.reconstruction, timeseries)
                    observed_mask = input_mask * (1 - mask)
                    masked_loss = observed_mask * recon_loss
                    loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)

                elif task_name == "detection":
                    timeseries, input_mask, label = data
                    n_channels = timeseries.shape[1]
                    seq_len = timeseries.shape[-1]
                    timeseries = timeseries.reshape(-1, 1, seq_len).float().to(self.device)
                    input_mask = input_mask.to(self.device).long()
                    input_mask = input_mask.repeat_interleave(n_channels, axis=0)
                    mask = mask_generator.generate_mask(x=timeseries, input_mask=input_mask).to(self.device).long()
                    output = self.model(x_enc=timeseries, input_mask=input_mask, mask=mask)
                    with torch.amp.autocast(device_type='cuda'):
                        loss = criterion(output.reconstruction, timeseries)
                    
                elif task_name == "classification":
                    timeseries, input_mask, label = data
                    timeseries = timeseries.to(self.device).float()
                    label = label.to(self.device).long()
                    output = self.model(x_enc=timeseries)
                    with torch.amp.autocast(device_type='cuda'):
                        loss = criterion(output.logits, label)

                optimizer.zero_grad(set_to_none=True)
                # Scales the loss for mixed precision training
                scaler.scale(loss).backward()

                # Clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()

                losses.append(loss.item())

            losses = np.array(losses)
            average_loss = np.average(losses)
            print(f"Epoch {epoch}: Train loss: {average_loss:.3f}")

            scheduler.step()

        return self.model
    
    def evaluate(self, dataset, task_name="forecasting"):
        dataloader = dataset.get_data_loader()
        criterion = torch.nn.MSELoss()
        self.model.to(self.device)
        self.model.eval()
        if task_name == "forecasting":
            trues, preds, histories, losses = [], [], [], []
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    # unpack the data
                    timeseries, input_mask, forecast  = data
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(self.device)
                    input_mask = input_mask.to(self.device)
                    forecast = forecast.float().to(self.device)

                    output = self.model(x_enc=timeseries, input_mask=input_mask)
                    loss = criterion(output.forecast, forecast)
                    losses.append(loss.item())
                    trues.append(forecast.detach().cpu().numpy())
                    preds.append(output.forecast.detach().cpu().numpy())
                    histories.append(timeseries.detach().cpu().numpy())

            losses = np.array(losses)
            average_loss = np.average(losses)
            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            histories = np.concatenate(histories, axis=0)

            return average_loss, trues, preds, histories
        
        elif task_name == "imputation":
            trues, preds, masks = [], [], []
            mask_generator = Masking(mask_ratio=0.25)
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    # unpack the data
                    timeseries, input_mask = data
                    trues.append(timeseries.numpy())
                    n_channels = timeseries.shape[1]
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(self.device)
                    timeseries = timeseries.reshape(-1, 1, timeseries.shape[-1])
                    # print(input_mask.shape)
                    input_mask = input_mask.to(self.device).long()
                    input_mask = input_mask.repeat_interleave(n_channels, axis=0)
                    # print(timeseries.shape, input_mask.shape)
                    mask = mask_generator.generate_mask(x=timeseries, input_mask=input_mask).to(self.device).long()
                    output = self.model(x_enc=timeseries, input_mask=input_mask, mask=mask)
                    reconstruction = output.reconstruction.reshape(-1, n_channels, timeseries.shape[-1])
                    mask = mask.reshape(-1, n_channels, timeseries.shape[-1])
                    preds.append(reconstruction.detach().cpu().numpy())
                    masks.append(mask.detach().cpu().numpy())

            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            masks = np.concatenate(masks, axis=0)

            return trues, preds, masks
        
        elif task_name == "detection":
            trues, preds, labels = [], [], []
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    # unpack the data
                    timeseries, input_mask, label = data
                    timeseries = timeseries.to(self.device).float()
                    input_mask = input_mask.to(self.device).long()
                    label = label.to(self.device).long()
                    output = self.model(x_enc=timeseries, input_mask=input_mask)

                    trues.append(timeseries.detach().cpu().numpy())
                    preds.append(output.reconstruction.detach().cpu().numpy())
                    labels.append(label.detach().cpu().numpy())

            trues = np.concatenate(trues, axis=0).flatten()
            preds = np.concatenate(preds, axis=0).flatten()
            labels = np.concatenate(labels, axis=0).flatten()

            return trues, preds, labels
        
        elif task_name == "classification":
            accuracy = 0
            total = 0
            embeddings = []
            labels = []
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    # unpack the data
                    timeseries, input_mask, label = data
                    timeseries = timeseries.to(self.device).float()
                    label = label.to(self.device).long()
                    labels.append(label.detach().cpu().numpy())
                    input_mask = input_mask.to(self.device).long()
                    output = self.model(x_enc=timeseries, input_mask=input_mask)
                    embedding = output.embeddings.mean(dim=1)
                    embeddings.append(embedding.detach().cpu().numpy())
                    _, predicted = torch.max(output.logits, 1)
                    total += label.size(0)
                    accuracy += (predicted == label).sum().item()

            accuracy = accuracy / total
            embeddings = np.concatenate(embeddings)
            labels = np.concatenate(labels)
            return accuracy, embeddings, labels



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

    tfm_model = TimesfmModel(config=config, repo=repo)
    model = tfm_model.model
    # print(tfm.model)
    df = pd.read_csv("/nethome/sli999/data/Tycho/dengue_laos.csv")
    df = df[df["SourceName"] == "Laos Dengue Surveillance System"]
    df = df[["Admin1ISO", "PeriodStartDate", "CountValue"]]
    df.columns = ["unique_id", "ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(by=["unique_id", "ds"])
    forecast_df = model.forecast_on_df(
        inputs=df,
        freq="D",  # daily frequency
        value_name="y",
        num_jobs=1,
    )
    forecast_df = forecast_df[["ds", "unique_id", "timesfm"]]
    forecast_df.columns = ["ds", "unique_id", "y"]

    print(forecast_df.head())
