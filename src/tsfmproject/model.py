import pandas as pd
import numpy as np
import torch
import logging
import glob
import os
import sys
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from .models.chronosforecasting.scripts.jsonlogger import JsonFileHandler, JsonFormatter


from .models.timesfm import timesfm as tfm
from .models.timesfm.timesfm import pytorch_patched_decoder as ppd
from .models.chronosforecasting.chronos import chronos
from chronos import ChronosPipeline
from .models.chronosforecasting.scripts import finetune

from .utils import get_least_used_gpu



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
        FinetuneModel.to(self.device)
        FinetuneModel.train()
        dataloader = dataloader.get_data_loader()
        optimizer = torch.optim.Adam(FinetuneModel.parameters(), lr=1e-4)
        epoch = 10 if "epoch" not in kwargs else kwargs["epoch"]
        avg_loss = 0
        for epoch in range(epoch):
            for i, (inputs) in enumerate(dataloader):
                inputs = dataloader.preprocess_train_batch(inputs)
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
    def __init__(self, config=None, repo=None):
        super().__init__(config=config, repo=repo)
        if self.config is None:
            self.config = {
                'context_length': 512,
                'prediction_length': 64,
                'min_past': 64,
                'max_steps': 100,
                'save_steps': 25,
                'log_steps': 5,
                'per_device_train_batch_size': 32,
                'learning_rate': 1e-3,
                'optim': 'adamw_torch_fused',
                'shuffle_buffer_length': 100,
                'gradient_accumulation_steps': 2,
                'model_id': 'amazon/chronos-t5-small',
                'model_type': 'seq2seq',
                'random_init': False,
                'tie_embeddings': False,
                'output_dir': os.path.join(sys.path[0],'./tsfmproject/models/chronosforecasting/output/finetuning/'),
                'tf32': True,
                'torch_compile': True,
                'tokenizer_class': 'MeanScaleUniformBins',
                'tokenizer_kwargs': {'low_limit': -15.0, 'high_limit': 15.0},
                'n_tokens': 4096,
                'n_special_tokens': 2,
                'pad_token_id': 0,
                'eos_token_id': 1,
                'use_eos_token': True,
                'lr_scheduler_type': 'linear',
                'warmup_ratio': 0.0,
                'dataloader_num_workers': 1,
                'max_missing_prop': 0.9,
                'num_samples': 10,
                'temperature': 1.0,
                'top_k': 50,
                'top_p': 1.0,
                'seed': 42
            }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.result_logger = self.setup_logger("results")
        self.evaluation_logger = self.setup_logger("evaluation")
        self.model = self.load_model(model_dir=repo)

    def setup_logger(self, log_type):
        log_dir = Path(os.path.join(sys.path[0],'./tsfmproject/models/chronosforecasting/output/') )/ log_type
        log_dir.mkdir(parents=True, exist_ok=True)

        log_files = sorted(log_dir.glob(f"{log_type}_*.json"), key=os.path.getmtime)
        if log_files:
            latest_file = log_files[-1]
            latest_index = int(latest_file.stem.split('_')[-1])
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

    def load_model(self, model_dir: str = "amazon/chronos-t5-small", model_type: str = "seq2seq"):
        self.model = ChronosPipeline.from_pretrained(model_dir, model_type=model_type)
        self.result_logger.info(f"Model loaded from {model_dir}")

    def get_latest_run_dir(self, base_dir=os.path.join(sys.path[0],"./tsfmproject/models/chronosforecasting/output/finetuning/")):
        run_dirs = glob.glob(os.path.join(base_dir, "run-*"))
        if not run_dirs:
            raise FileNotFoundError("No run directories found.")
        latest_run_dir = max(run_dirs, key=os.path.getmtime)
        return latest_run_dir

    def finetune(self, dataset, probability_list=None, **kwargs):
        
        # Convert dataset to arrow format
        data_loc = os.path.join(sys.path[0],'./tsfmproject/models/chronosforecasting/data/data.arrow')
        time_series_list = [np.array(dataset.dataset[column].values) for column in dataset.ts_cols]
        dataset.convert_to_arrow(data_loc, time_series=time_series_list, start_date=dataset.dataset.index[0])
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
            **finetune_config
        )

    def evaluate(self, dataset, metrics=['MSE'], **kwargs):
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
        data = dataset.dataset
        context_len = dataset.context_len
        horizon_len = dataset.horizon_len
        total_len = context_len + horizon_len
        stride = kwargs.get('stride', 1)
        quantiles = kwargs.get('quantiles', [0.1, 0.5, 0.9])

        eval_windows = []
        true_values = []
        predictions = []
        histories = []
        means = []

        for start in range(0, len(data) - total_len + 1, stride):
            window = data[start:start + total_len]
            context = window[:context_len].to_numpy().transpose()
            actual = window[context_len:].to_numpy().transpose()
            input = [torch.tensor(ts) for i, ts in enumerate(context)]
            prediction = self.model.predict(context=input, prediction_length=horizon_len, num_samples=20)
            pred_median = np.median(prediction, axis=1)
            pred_values = np.quantile(prediction, q=[0.1,0.5,0.9], axis=1).transpose(1,0,2)
            pred_values = pred_values.squeeze()
            # pred_values = pred_values.permute(0, 2, 1).numpy().squeeze()
            # mean1 = mean.numpy().squeeze()
            eval = {}
            for metric in metrics:
                if metric == 'MSE':
                    # flatten the arrays

                    eval[metric] = np.mean((actual - pred_median) ** 2).item()
                elif metric == 'MAPE':
                    eval[metric] = mean_absolute_percentage_error(actual.flatten(), mean1.flatten())
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
            eval_windows.append(eval)
            true_values.append(actual)
            predictions.append(pred_values)
            histories.append(context)

        # get average evaluation results from all windows
        eval_results = {}
        for metric in metrics:
            eval_results[metric] = np.mean([eval[metric] for eval in eval_windows])


        return np.array(eval_results), np.array(true_values), np.array(predictions), np.array(histories)

    def forecast(self, input, **kwargs):
        context = torch.tensor(input)
        prediction_length = kwargs.get('prediction_length', 64)
        predictions = self.model.predict(context, prediction_length=prediction_length).squeeze()
        pred_values = np.quantile(predictions.numpy(), [0.5, 0.1, 0.9], axis=-2)
        return predictions, pred_values


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
