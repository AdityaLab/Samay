import pandas as pd
import torch
import logging
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
            'output_dir': './src/tsfmproject/models/chronosforecasting/output/finetuning/',
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
        self.model = None
        self.logger = self.setup_logger()

    def setup_logger(self):
        log_file = Path("evaluation_results.json")
        json_handler = JsonFileHandler(log_file)
        json_handler.setFormatter(JsonFormatter())

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(json_handler)
        return logger

    def load_model(self, model_dir: str = "amazon/chronos-t5-small", model_type: str = "seq2seq"):
        self.model = ChronosPipeline.from_pretrained(model_dir, model_type=model_type)
        self.logger.info(f"Model loaded from {model_dir}")

    def finetune(self, training_data_paths, probability_list=None, **kwargs):
        # Use default probability_list if None
        if probability_list is None:
            probability_list = [1]

        # Merge provided kwargs with default configuration
        finetune_config = self.config.copy()
        # Update with kwargs where values are not None
        finetune_config.update({k: v for k, v in kwargs.items() if v is not None})

        # Call the train_model function with the combined configuration
        finetune.train_model(
            training_data_paths=training_data_paths,
            probability=probability_list,
            logger=self.logger,
            **finetune_config
        )

    def evaluate(self, fit_data, test_data, prediction_length, metrics):
        context = torch.tensor(fit_data)
        predictions = self.model.predict(context, prediction_length=prediction_length).squeeze().tolist()

        results = {}
        results['num_samples'] = len(predictions)
        results['predictions'] = predictions[0]

        if 'RMSE' in metrics:
            results['RMSE'] = mean_squared_error(test_data, predictions[0], squared=False)
        if 'MAPE' in metrics:
            results['MAPE'] = mean_absolute_percentage_error(test_data, predictions[0])

        self.logger.info(f"Evaluation results: {results}")

        return results

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
