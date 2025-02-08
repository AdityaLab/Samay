import glob
import logging
import os
import sys
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import torch
# from chronos import ChronosPipeline
from samay.models.chronosforecasting.chronos.chronos import ChronosPipeline
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from samay.models.chronosforecasting.scripts import finetune
from samay.models.chronosforecasting.scripts.jsonlogger import JsonFileHandler, JsonFormatter
from samay.models.moment.momentfm.models.moment import MOMENTPipeline
from samay.models.moment.momentfm.utils.masking import Masking
from samay.models.timesfm import timesfm as tfm
from samay.models.timesfm.timesfm import pytorch_patched_decoder as ppd
from samay.models.uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from samay.models.uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from samay.models.uni2ts.model.moirai.finetune import MoiraiFinetune
from samay.dataset import MoiraiDataset
from samay.utils import get_least_used_gpu


class Basemodel:
    def __init__(self, config=None, repo=None):
        """
        Args:
            config: dict, model configuration
            repo: str, Huggingface model repository id
        """
        self.config = config
        self.repo = repo
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
        lr = 1e-4 if "lr" not in kwargs else kwargs["lr"]
        epoch = 10 if "epoch" not in kwargs else kwargs["epoch"]

        core_layer_tpl = self.model._model
        # Todo: whether add freq
        FinetunedModel = ppd.PatchedDecoderFinetuneModel(core_layer_tpl=core_layer_tpl)
        if freeze_transformer:
            for param in FinetunedModel.core_layer.stacked_transformer.parameters():
                param.requires_grad = False
        FinetunedModel.to(self.device)
        FinetunedModel.train()
        dataloader = dataset.get_data_loader()
        optimizer = torch.optim.Adam(FinetunedModel.parameters(), lr=lr)

        avg_loss = 0
        for epoch in range(epoch):
            for i, (inputs) in enumerate(dataloader):
                inputs = dataset.preprocess(inputs)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                optimizer.zero_grad()
                outputs = FinetunedModel.compute_predictions(
                    inputs
                )  # b, n, seq_len, 1+quantiles
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
                input_ts = np.squeeze(input_ts, axis=0)
                actual_ts = inputs["actual_ts"].detach().cpu().numpy()
                actual_ts = np.squeeze(actual_ts, axis=0)

                output, _ = self.model.forecast(input_ts)
                output = output[:, 0 : actual_ts.shape[1]]

                loss = np.mean((output - actual_ts) ** 2)
                losses.append(loss.item())
                trues.append(actual_ts)
                preds.append(output)
                histories.append(input_ts)

        losses = np.array(losses)
        average_loss = np.average(losses)
        trues = np.concatenate(trues, axis=0).reshape(
            -1, dataset.num_ts, trues[-1].shape[-1]
        )
        preds = np.concatenate(preds, axis=0).reshape(
            -1, dataset.num_ts, preds[-1].shape[-1]
        )
        histories = np.concatenate(histories, axis=0).reshape(
            -1, dataset.num_ts, histories[-1].shape[-1]
        )

        return average_loss, trues, preds, histories


class ChronosModel(Basemodel):
    def __init__(self, config=None, repo=None):
        super().__init__(config=config, repo=repo)
        if self.config is None:
            self.config = {
                "context_length": 512,
                "prediction_length": 64,
                "min_past": 64,
                "max_steps": 100,
                "save_steps": 25,
                "log_steps": 5,
                "per_device_train_batch_size": 32,
                "learning_rate": 1e-3,
                "optim": "adamw_torch_fused",
                "shuffle_buffer_length": 100,
                "gradient_accumulation_steps": 2,
                "model_id": "amazon/chronos-t5-small",
                "model_type": "seq2seq",
                "random_init": False,
                "tie_embeddings": False,
                "output_dir": os.path.join(
                    sys.path[0],
                    "./tsfmproject/models/chronosforecasting/output/finetuning/",
                ),
                "tf32": True,
                "torch_compile": True,
                "tokenizer_class": "MeanScaleUniformBins",
                "tokenizer_kwargs": {"low_limit": -15.0, "high_limit": 15.0},
                "n_tokens": 4096,
                "n_special_tokens": 2,
                "pad_token_id": 0,
                "eos_token_id": 1,
                "use_eos_token": True,
                "lr_scheduler_type": "linear",
                "warmup_ratio": 0.0,
                "dataloader_num_workers": 1,
                "max_missing_prop": 0.9,
                "num_samples": 10,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 1.0,
                "seed": 42,
            }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cuda")
        self.result_logger = self.setup_logger("results")
        self.evaluation_logger = self.setup_logger("evaluation")
        self.model = self.load_model(model_dir=self.repo, model_type="seq2seq")

    def setup_logger(self, log_type):
        log_dir = (
            Path(
                os.path.join(
                    sys.path[0], "./tsfmproject/models/chronosforecasting/output/"
                )
            )
            / log_type
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        log_files = sorted(log_dir.glob(f"{log_type}_*.json"), key=os.path.getmtime)
        if log_files:
            latest_file = log_files[-1]
            latest_index = int(latest_file.stem.split("_")[-1])
            new_index = latest_index + 1
        else:
            new_index = 1

        log_file = log_dir / f"{log_type}_{new_index}.json"
        json_handler = JsonFileHandler(log_file)
        json_handler.setFormatter(JsonFormatter(log_type))

        logger = logging.getLogger(f"{log_type}_logger")
        logger.setLevel(logging.INFO)
        logger.addHandler(json_handler)
        return logger

    def load_model(
        self, model_dir: str = "amazon/chronos-t5-small", model_type: str = "seq2seq"
    ):
        self.model = ChronosPipeline.from_pretrained(
            model_dir,
            model_type=model_type,
            device_map=self.device,
            torch_dtype=torch.float32,
        )
        self.result_logger.info(f"Model loaded from {model_dir}")

    def get_latest_run_dir(
        self,
        base_dir=os.path.join(
            sys.path[0], "./tsfmproject/models/chronosforecasting/output/finetuning/"
        ),
    ):
        run_dirs = glob.glob(os.path.join(base_dir, "run-*"))
        if not run_dirs:
            raise FileNotFoundError("No run directories found.")
        latest_run_dir = max(run_dirs, key=os.path.getmtime)
        return latest_run_dir

    def finetune(self, dataset, probability_list=None, **kwargs):
        # Convert dataset to arrow format
        data_loc = os.path.join(
            sys.path[0], "./tsfmproject/models/chronosforecasting/data/data.arrow"
        )

        time_series_list = [
            np.array(dataset.dataset[column].values) for column in dataset.ts_cols
        ]
        dataset.convert_to_arrow(
            data_loc, time_series=time_series_list, start_date=dataset.dataset.index[0]
        )
        # Use default probability_list if None
        if probability_list is None:
            probability_list = [1]

        # Merge provided kwargs with default configuration
        finetune_config = self.config.copy()
        # Update with kwargs where values are not None
        finetune_config.update({k: v for k, v in kwargs.items() if v is not None})

        # Call the train_model function with the combined configuration
        finetune.train_model(
            training_data_paths=[data_loc],
            probability=probability_list,
            logger=self.result_logger,
            **finetune_config,
        )

    def evaluate(self, dataset, metrics=["MSE"], **kwargs):
        """
        Evaluate the model on the given train and test data.

        Args:
            train_data (pd.DataFrame): The training data.
            test_data (pd.DataFrame): The testing data.
            offset (int): The offset for slicing the data.
            metrics (list): List of metrics to evaluate.

        Returns:
            dict: Evaluation results for each column.
            dict: True values for each column.
            dict: Predictions for each column.
            dict: Histories for each column.
        """
        # data = dataset.dataset
        context_len = dataset.context_len
        horizon_len = dataset.horizon_len
        total_len = context_len + horizon_len
        quantiles = kwargs.get("quantiles", [0.1, 0.5, 0.9])
        batch_size = kwargs.get("batch_size", 8)

        dataloader = DataLoader(
            dataset.dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        eval_windows = []
        true_values = []
        predictions = []
        histories = []

        # self.model.to('cuda')  # Move model to GPU
        with torch.no_grad():
            for i, (history, actual) in enumerate(dataloader):
                # context = context.to('cuda')
                # actual = actual.to('cuda')
                actual = actual.detach().cpu().numpy()
                history = history
                history_stack = history.reshape(-1, context_len)
                prediction = (
                    self.model.predict(
                        context=history_stack,
                        prediction_length=horizon_len,
                        num_samples=20,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                pred_median = np.median(prediction, axis=1)
                pred_median = pred_median.reshape(
                    actual.shape[0], actual.shape[1], horizon_len
                )

                # pred_median = np.median(prediction, axis=1)
                # pred_values = np.quantile(prediction, q=quantiles, axis=1).transpose(1, 0, 2).squeeze()
                # pred_values = prediction.squeeze().numpy()

                eval = []
                for metric in metrics:
                    if metric == "MSE":
                        eval.append(np.mean((actual - pred_median) ** 2))
                    elif metric == "MASE":
                        forecast_error = np.mean(np.abs(actual - pred_median))
                        naive_error = np.mean(
                            np.abs(actual[:, :, 1:] - actual[:, :, :-1])
                        )
                        if naive_error == 0:
                            eval.append(np.inf)
                        else:
                            eval.append(forecast_error / naive_error)

                    else:
                        raise ValueError(f"Unsupported metric: {metric}")

                eval_windows.append(eval)
                true_values.append(actual)
                predictions.append(pred_median)
                histories.append(history)

        true_values = np.concatenate(true_values, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        histories = np.concatenate(histories, axis=0)

        # get average evaluation results from all windows
        eval_windows = np.mean(np.array(eval_windows), axis=0)
        eval_results = {}
        for i in range(len(metrics)):
            eval_results[metrics[i]] = eval_windows[i]

        return eval_results, true_values, predictions, histories

    def forecast(self, input, **kwargs):
        context = torch.tensor(input)
        prediction_length = kwargs.get("prediction_length", 64)
        predictions = self.model.predict(
            context, prediction_length=prediction_length
        ).squeeze()
        pred_values = np.quantile(predictions.numpy(), [0.5, 0.1, 0.9], axis=-2)
        return predictions, pred_values


class MomentModel(Basemodel):
    def __init__(self, config=None, repo=None):
        super().__init__(config=config, repo=repo)
        if not repo:
            raise ValueError("Moment model requires a repository")
        self.model = MOMENTPipeline.from_pretrained(repo, model_kwargs=self.config)
        self.model.init()

    def finetune(self, dataset, task_name="forecasting", **kwargs):
        # arguments
        max_lr = 1e-4 if "lr" not in kwargs else kwargs["lr"]
        max_epoch = 5 if "epoch" not in kwargs else kwargs["epoch"]
        max_norm = 5.0 if "norm" not in kwargs else kwargs["norm"]
        mask_ratio = 0.25 if "mask_ratio" not in kwargs else kwargs["mask_ratio"]

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
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3
        )
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
                    with torch.amp.autocast(device_type="cuda"):
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
                    mask = (
                        mask_generator.generate_mask(
                            x=timeseries, input_mask=input_mask
                        )
                        .to(self.device)
                        .long()
                    )
                    output = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=mask
                    )
                    with torch.amp.autocast(device_type="cuda"):
                        recon_loss = criterion(output.reconstruction, timeseries)
                    observed_mask = input_mask * (1 - mask)
                    masked_loss = observed_mask * recon_loss
                    loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)

                elif task_name == "detection":
                    timeseries, input_mask, label = data
                    n_channels = timeseries.shape[1]
                    seq_len = timeseries.shape[-1]
                    timeseries = (
                        timeseries.reshape(-1, 1, seq_len).float().to(self.device)
                    )
                    input_mask = input_mask.to(self.device).long()
                    input_mask = input_mask.repeat_interleave(n_channels, axis=0)
                    mask = (
                        mask_generator.generate_mask(
                            x=timeseries, input_mask=input_mask
                        )
                        .to(self.device)
                        .long()
                    )
                    output = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=mask
                    )
                    with torch.amp.autocast(device_type="cuda"):
                        loss = criterion(output.reconstruction, timeseries)

                elif task_name == "classification":
                    timeseries, input_mask, label = data
                    timeseries = timeseries.to(self.device).float()
                    label = label.to(self.device).long()
                    output = self.model(x_enc=timeseries)
                    with torch.amp.autocast(device_type="cuda"):
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
                    timeseries, input_mask, forecast = data
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
                    mask = (
                        mask_generator.generate_mask(
                            x=timeseries, input_mask=input_mask
                        )
                        .to(self.device)
                        .long()
                    )
                    output = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=mask
                    )
                    reconstruction = output.reconstruction.reshape(
                        -1, n_channels, timeseries.shape[-1]
                    )
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


class MoiraiTSModel(Basemodel):
    def __init__(
        self,
        config=None,
        repo=None,
        model_type="moirai-moe",
        model_size="small",
        **kwargs,
    ):
        super().__init__(config=config, repo=repo)
        self.horizon_len = config.get("horizon_len", 32)
        self.context_len = config.get("context_len", 128)
        self.patch_size = config.get("patch_size", 16)
        self.batch_size = config.get("batch_size", 16)
        self.num_samples = config.get("num_samples", 100)
        self.target_dim = config.get("target_dim", 1)
        self.feat_dynamic_real_dim = config.get("feat_dynamic_real_dim", 0)
        self.past_feat_dynamic_real_dim = config.get("past_feat_dynamic_real_dim", 0)
        self.model_type = model_type

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == "moirai": # standard moirai
            if self.repo is None:
                self.repo = f"Salesforce/moirai-1.1-R-{model_size}"
            self.model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(self.repo),
                prediction_length=self.horizon_len,
                context_length=self.context_len,
                patch_size=self.patch_size,
                num_samples=self.num_samples,
                target_dim=self.target_dim,
                feat_dynamic_real_dim=self.feat_dynamic_real_dim,
                past_feat_dynamic_real_dim=self.past_feat_dynamic_real_dim,
            )
        elif model_type == "moirai-moe": # moirai with Mixture of Experts
            if self.repo is None:
                self.repo = f"Salesforce/moirai-moe-1.0-R-{model_size}"
            self.model = MoiraiMoEForecast(
                module=MoiraiMoEModule.from_pretrained(self.repo),
                prediction_length=self.horizon_len,
                context_length=self.context_len,
                patch_size=self.patch_size,
                num_samples=self.num_samples,
                target_dim=self.target_dim,
                feat_dynamic_real_dim=self.feat_dynamic_real_dim,
                past_feat_dynamic_real_dim=self.past_feat_dynamic_real_dim,
            )
        self.model.to(self.device)

    def evaluate(self, dataset:MoiraiDataset, metrics=["MSE"], **kwargs):
        """For a given test dataset, we evaluate the model using the given metrics.

        Args:
            dataset (MoiraiDataset): Dataset to evaluate the model on.
            metrics (list, optional): Metrics you want to evaluate the model on. Defaults to ["MSE"].

        Raises:
            ValueError: Any metric other than "MSE" or "MASE is not supported.

        Returns:
            dict: Evaluation results for each column (variate).
            dict: True values for each column (variate).
            dict: Predictions for each column (variate).
            dict: Histories for each column (variate).
        """
        predictor = self.model.create_predictor(batch_size=self.batch_size)
        forecast = predictor.predict(dataset.dataset.input)

        # Iterators for input, label and forecast
        input_it = iter(dataset.dataset.input)
        label_it = iter(dataset.dataset.label)
        forecast_it = iter(forecast)

        trues = {}
        preds = {}
        histories = {}
        eval_windows = []

        with torch.no_grad(): # No need to compute gradients

            # Iterate over each window
            for input, label, forecast in zip(input_it, label_it, forecast_it):
                true_values = np.array(label["target"])
                past_values = np.array(input["target"])
                pred_values = np.median(forecast.samples, axis=0)
                length = len(past_values)

                eval = []
                for metric in metrics:
                    if metric == "MSE":
                        eval.append(mean_squared_error(true_values, pred_values))
                    
                    # MASE = current model's MAE / naive model's MAE
                    elif metric == "MASE":
                        forecast_error = np.mean(np.abs(true_values - pred_values))
                        naive_error = np.mean(np.abs(true_values[1:] - true_values[:-1]))
                        if naive_error == 0: # Avoid division by zero
                            eval.append(np.inf)
                        else:
                            eval.append(forecast_error / naive_error)
                    else:
                        raise ValueError(f"Unsupported metric: {metric}")
                eval_windows.append(eval)

                # Update history, true values and predictions
                if length not in histories.keys():
                    histories[length] = []
                    trues[length] = []
                    preds[length] = []
                histories[length].append(past_values)
                trues[length].append(true_values)
                preds[length].append(pred_values)

        eval_windows = np.mean(np.array(eval_windows), axis=0)
        eval_results = {}
        for i in range(len(metrics)):
            eval_results[metrics[i]] = eval_windows[i]

        # Convert to numpy arrays
        histories = [np.array(histories[key]) for key in histories.keys()]
        trues = [np.array(trues[key]) for key in trues.keys()]
        preds = [np.array(preds[key]) for key in preds.keys()]

        return eval_results, trues, preds, histories
    
    def finetune(self, dataset, **kwargs):

        # Parameters
        path = "../src/samay/models/uni2ts/cli/conf/finetune/model/moirai_small.yaml"
        with open(path, "r") as file:
            fin_config = yaml.safe_load(file)
        epochs = 10 if "epochs" not in kwargs else kwargs["epochs"]
        num_batches = len(dataset.dataset)//self.batch_size
        num_batches_per_epoch = num_batches//epochs
        training_steps = num_batches_per_epoch * epochs
        
        # Load the model
        FinetunedModel = MoiraiFinetune(min_patches=fin_config["min_patches"],
                                        min_mask_ratio=fin_config["min_mask_ratio"],
                                        max_mask_ratio=fin_config["max_mask_ratio"],
                                        max_dim=fin_config["max_dim"],
                                        num_training_steps=training_steps,
                                        num_warmup_steps=fin_config["num_warmup_steps"],
                                        module_kwargs=fin_config["module_kwargs"],
                                        beta1=fin_config["beta1"],
                                        beta2=fin_config["beta2"],
                                        val_metric=fin_config["val_metric"],
                                        weight_decay=fin_config["weight_decay"]
                                        )
        
        # No need to worry about freezing layers
        # MOIRAI's finetune.py already takes care of it - blacklist_params (line 259, finetune.py)
        FinetunedModel.to(self.device)
        FinetunedModel.train() # Set model to training mode

        # Load the dataset
        dataloader = dataset.get_data_loader()
        

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
