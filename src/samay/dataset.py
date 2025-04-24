import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split as ts_split
from gluonts.transform import (
    AddObservedValuesIndicator,
    AsNumpyArray,
)
from pandas._libs.tslibs.period import Period
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from .models.chronosforecasting.chronos.chronos import (
    ChronosConfig,
    MeanScaleUniformBins,
)
from .models.timesfm.timesfm.data_loader import TimeSeriesdata
from .moirai_utils import (
    AddObservedValues,
    ArrExpandDims,
    AsNumpy,
    CausalMeanNaNFix,
    MoiraiTorch,
    custom_train_instance_split,
)
from .utils import get_multivariate_data


# function for specific dataset to download and preprocess data, returning path
# BaseDataset class call the specific function decided by "name" argument
class BaseDataset:
    def __init__(
        self,
        name=None,
        datetime_col=None,
        path=None,
        batchsize=8,
        mode="train",
        **kwargs,
    ):
        """
        Args:
            name: str, dataset name
            target: np.ndarray, target data
        """
        self.name = name
        self.datetime_col = datetime_col
        self.batchsize = batchsize
        self.mode = mode
        if path:
            self.data_path = path
        else:
            data_func = globals()[f"get_{self.name}_dataset"]
            self.data_path = data_func()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dt = self.data[idx]
        dt = self.preprocess(dt)
        return dt

    def preprocess(self, **kwargs):
        raise NotImplementedError

    def get_data_loader(self):
        raise NotImplementedError

    def save(self, path):
        save_path = path
        torch.save(self.data, save_path)


def get_tycho_dataset():
    """
    Download and preprocess Tycho dataset
    Returns:
        data_path: str, path to the preprocessed data
    """
    repo_id = "username/tycho"
    # download data
    data = load_dataset(repo_id, cache_dir="data/Tycho")
    data_path = "data/Tycho/Tycho.csv"

    return data_path


def get_ett_dataset():
    """
    Download and preprocess ETTh dataset
    Returns:
        data_path: str, path to the preprocessed data
    """
    repo_id = "username/ett"
    # download data
    data = load_dataset(repo_id, cache_dir="data/ETTh")
    data_path = "data/ETTh/ETTh.csv"

    return data_path


def get_ecg5000_dataset():
    """
    Download and preprocess ECG5000 dataset
    Returns:
        data_path: str, path to the preprocessed data
    """
    repo_id = "username/ECG5000"
    # download data
    data = load_dataset(repo_id, cache_dir="data/ECG5000")
    data_path = "data/ECG5000/ECG5000.csv"

    return data_path


def get_tiltABP2_dataset():
    """
    Download and preprocess tiltABP2 dataset
    Returns:
        data_path: str, path to the preprocessed data
    """
    repo_id = "username/tiltABP2"
    # download data
    data = load_dataset(repo_id, cache_dir="data/tiltABP2")
    data_path = "data/tiltABP2/tiltABP2.csv"

    return data_path


class TimesfmDataset(BaseDataset):
    """
    Dataset class for TimesFM model
    Data Format:
    Dict with keys:
    input_ts: np.ndarray, historical time series data
    actual_ts: np.ndarray, actual time series data
    """

    def __init__(
        self,
        name=None,
        datetime_col="ds",
        path=None,
        batchsize=4,
        mode="train",
        boundaries=(0, 0, 0),
        context_len=128,
        horizon_len=32,
        freq="h",
        normalize=False,
        stride=10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            datetime_col=datetime_col,
            path=path,
            batchsize=batchsize,
            mode=mode,
        )
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.freq = freq
        self.normalize = normalize
        self.stride = stride
        self.data = pd.read_csv(self.data_path)
        if boundaries == (0, 0, 0):
            # Default boundaries: train 50%, val 20%, test 30%
            self.boundaries = [
                int(len(self.data) * 0.5),
                int(len(self.data) * 0.7),
                len(self.data) - 1,
            ]
        elif boundaries == (-1, -1, -1):
            # use all data for training
            self.boundaries = [0, 0, len(self.data) - 1]
        else:
            self.boundaries = boundaries
        self.horizon_len = min(self.horizon_len, int(0.3 * len(self.data) + 1))
        self.ts_cols = [col for col in self.data.columns if col != self.datetime_col]
        tfdtl = TimeSeriesdata(
            data_path=self.data_path,
            datetime_col=self.datetime_col,
            num_cov_cols=None,
            cat_cov_cols=None,
            ts_cols=np.array(self.ts_cols),
            train_range=[0, self.boundaries[0]],
            val_range=[self.boundaries[0], self.boundaries[1]],
            test_range=[self.boundaries[1], self.boundaries[2]],
            hist_len=self.context_len,
            pred_len=self.horizon_len,
            batch_size=64,
            freq=self.freq,
            normalize=self.normalize,
            epoch_len=None,
            holiday=False,
            permute=False,
        )
        self.num_ts = len(self.ts_cols)
        if self.mode == "train":
            tfset = tfdtl.torch_dataset(mode="train", shift=self.stride)
        else:
            tfset = tfdtl.torch_dataset(mode="test", shift=self.horizon_len)
        self.dataset = tfset

    def get_data_loader(self):
        if self.mode == "train":
            return DataLoader(self.dataset, batch_size=self.batchsize, shuffle=True)
        else:
            return DataLoader(self.dataset, shuffle=False)

    def preprocess_train_batch(self, data):
        past_ts = data[0].reshape(data[0].shape[0] * data[0].shape[1], -1)
        actual_ts = data[3].reshape(data[3].shape[0] * data[3].shape[1], -1)
        return {"input_ts": past_ts, "actual_ts": actual_ts}

    def preprocess_eval_batch(self, data):
        past_ts = data[0]
        actual_ts = data[3]
        return {"input_ts": past_ts, "actual_ts": actual_ts}

    def preprocess(self, data):
        if self.mode == "train":
            return self.preprocess_train_batch(data)
        else:
            return self.preprocess_eval_batch(data)


class ChronosDataset(BaseDataset):
    """
    Dataset class for Chronos model
    Data Format:
    Dict with keys:
    input_ts: np.ndarray, historical time series data
    actual_ts: np.ndarray, actual time series data
    """

    def __init__(
        self,
        name=None,
        datetime_col="ds",
        path=None,
        boundaries=[0, 0, 0],
        batch_size=16,
        mode=None,
        stride=10,
        tokenizer_class="MeanScaleUniformBins",
        drop_prob=0.2,
        min_past=64,
        np_dtype=np.float32,
        config=None,
    ):
        super().__init__(
            name=name,
            datetime_col=datetime_col,
            path=path,
            batchsize=batch_size,
            mode=mode,
        )
        # Todo: implement ChronosDataset
        assert tokenizer_class is not None, "Tokenizer is required for ChronosDataset"

        if not config:
            self.config = ChronosConfig(
                tokenizer_class="MeanScaleUniformBins",
                tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
                n_tokens=4096,
                n_special_tokens=2,
                pad_token_id=0,
                eos_token_id=1,
                use_eos_token=True,
                model_type="seq2seq",
                context_length=512,
                prediction_length=64,
                num_samples=20,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
            )
        else:
            self.config = ChronosConfig(**config)
        assert type(self.config) == ChronosConfig, (
            "Config must be an instance of ChronosConfig"
        )
        assert self.config.model_type in ("seq2seq", "causal"), (
            "Model type must be either 'seq2seq' or 'causal'"
        )

        self.context_len = self.config.context_length
        self.horizon_len = self.config.prediction_length
        self.drop_prob = drop_prob if self.config.model_type == "seq2seq" else 0.0
        self.min_past = min_past or self.config.prediction_length
        self.model_type = self.config.model_type
        self.mode = mode
        self.np_dtype = np_dtype
        self.boundaries = boundaries
        self.stride = stride
        self.batchsize = batch_size
        self.max_col_num = 16

        self.pad = False
        self._read_data()
        self.preprocess()

        self.one_chunk_num = (
            self.length_timeseries - self.context_len - self.horizon_len
        ) // self.stride + 1

    def _read_data(self):
        self.df = pd.read_csv(self.data_path)

        if self.boundaries[0] == 0:
            self.boundaries[0] = int(len(self.df) * 0.5)
        if self.boundaries[1] == 0:
            self.boundaries[1] = int(len(self.df) * 0.7)
        if self.boundaries[2] == 0:
            self.boundaries[2] = int(len(self.df) - 1)

        if self.boundaries == [-1, -1, -1]:
            # use all data for training
            self.boundaries = [0, 0, len(self.df) - 1]

        self.horizon_len = min(self.horizon_len, int(0.3 * len(self.df) + 1))

        self.n_channels = self.df.shape[1] - 1
        self.num_chunks = (self.n_channels + self.max_col_num - 1) // self.max_col_num

        if self.datetime_col:
            self.df.drop(columns=[self.datetime_col], inplace=True)

        self.df = np.array(self.df)

        if self.mode == "train":
            self.data = self.df[slice(0, self.boundaries[0]), :]

        elif self.mode == "test":
            self.data = self.df[slice(self.boundaries[1], self.boundaries[2]), :]

        self.length_timeseries = self.data.shape[0]
        self.required_len = self.context_len + self.horizon_len
        self.pad_len = 0
        if self.length_timeseries < self.required_len:
            self.pad = True
        self.pad_sequence()

    def pad_sequence(self):
        self.pad_len = self.required_len - self.length_timeseries
        # Pad data with zeros from the left
        if self.pad:
            self.data = np.pad(self.data, ((self.pad_len, 0), (0, 0)))
        # If num of channels isn't multiple of max_col_num, pad with zeros
        if (
            self.n_channels % self.max_col_num != 0
            and self.n_channels > self.max_col_num
        ):
            self.data = np.pad(
                self.data,
                ((0, 0), (0, self.max_col_num - self.n_channels % self.max_col_num)),
            )
        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        chunk_index = index // self.one_chunk_num
        data_chunk = (
            self.data[
                :, chunk_index * self.max_col_num : (chunk_index + 1) * self.max_col_num
            ]
            if (chunk_index + 1) * self.max_col_num < self.n_channels
            else self.data[:, chunk_index * self.max_col_num :]
        )
        seq_start = self.stride * (index % self.one_chunk_num)
        seq_end = seq_start + self.context_len
        input_mask = np.ones(self.context_len)
        # if the sequence is padded, mask of padded part is 0
        input_mask[: self.pad_len] = 0

        pred_end = seq_end + self.horizon_len

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = pred_end - self.horizon_len
            seq_start = seq_end - self.context_len

        self.config.prediction_length = self.horizon_len
        self.tokenizer = MeanScaleUniformBins(
            **self.config.tokenizer_kwargs, config=self.config
        )
        # input_seq = self.data[seq_start:seq_end, :].T
        input_seq = data_chunk[seq_start:seq_end, :].T
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(
            torch.tensor(input_seq)
        )
        forecast_seq = data_chunk[seq_end:pred_end, :].T
        labels, labels_mask = self.tokenizer.label_input_transform(
            torch.tensor(forecast_seq), scale
        )
        labels[labels_mask == 0] = -100
        return {
            "input_seq": input_seq,
            "forecast_seq": forecast_seq,
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    def __len__(self):
        if self.length_timeseries < self.context_len + self.horizon_len:
            return 1 * self.num_chunks
        return self.num_chunks * self.one_chunk_num

    def get_data_loader(self):
        if self.mode == "train":
            # dtl = DataLoader(self, batch_size=self.batchsize, shuffle=True)
            # for i, data in enumerate(dtl):
            #     timeseries, input_mask, forecast = data
            #     print(self.data.shape)
            #     print(timeseries.shape, input_mask.shape, forecast.shape)
            #     break
            return DataLoader(self, shuffle=True, batch_size=self.batchsize)
        else:
            return DataLoader(self, shuffle=False, batch_size=self.batchsize)

    def preprocess(self):
        if self.mode == "train" and self.drop_prob > 0:
            target = self.data.copy()
            drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
            mask = np.random.choice(
                [True, False], size=target.shape, p=[drop_p, 1 - drop_p]
            )
            target[mask] = np.nan
            self.data = target


class ChronosBoltDataset(BaseDataset):
    """
    Dataset class for ChronosBolt model
    Data Format:
    Dict with keys:
    input_ts: np.ndarray, historical time series data
    actual_ts: np.ndarray, actual time series data
    """

    def __init__(
        self,
        name=None,
        datetime_col="ds",
        path=None,
        boundaries=[0, 0, 0],
        batch_size=16,
        mode=None,
        stride=10,
        context_len=512,
        horizon_len=64,
    ):
        super().__init__(
            name=name,
            datetime_col=datetime_col,
            path=path,
            batchsize=batch_size,
            mode=mode,
        )
        # Todo: implement ChronosDataset

        self.context_len = context_len
        self.horizon_len = horizon_len
        self.mode = mode
        self.boundaries = boundaries
        self.stride = stride
        self.batchsize = batch_size
        self.max_col_num = 64

        self.pad = False
        self._read_data()

        self.one_chunk_num = (
            self.length_timeseries - self.context_len - self.horizon_len
        ) // self.stride + 1

    def _read_data(self):
        self.df = pd.read_csv(self.data_path)

        if self.boundaries[0] == 0:
            self.boundaries[0] = int(len(self.df) * 0.5)
        if self.boundaries[1] == 0:
            self.boundaries[1] = int(len(self.df) * 0.7)
        if self.boundaries[2] == 0:
            self.boundaries[2] = int(len(self.df) - 1)

        if self.boundaries == [-1, -1, -1]:
            # use all data for training
            self.boundaries = [0, 0, len(self.df) - 1]

        self.horizon_len = min(self.horizon_len, int(0.3 * len(self.df) + 1))

        self.n_channels = self.df.shape[1] - 1
        self.num_chunks = (self.n_channels + self.max_col_num - 1) // self.max_col_num

        if self.datetime_col:
            self.df.drop(columns=[self.datetime_col], inplace=True)

        self.df = np.array(self.df)

        if self.mode == "train":
            self.data = self.df[slice(0, self.boundaries[0]), :]

        elif self.mode == "test":
            self.data = self.df[slice(self.boundaries[1], self.boundaries[2]), :]

        self.length_timeseries = self.data.shape[0]
        self.required_len = self.context_len + self.horizon_len
        self.pad_len = 0
        if self.length_timeseries < self.required_len:
            self.pad = True
        self.pad_sequence()

    def pad_sequence(self):
        self.pad_len = self.required_len - self.length_timeseries
        # Pad data with zeros from the left
        if self.pad:
            self.data = np.pad(self.data, ((self.pad_len, 0), (0, 0)))
        # If num of channels isn't multiple of max_col_num, pad with zeros
        if (
            self.n_channels % self.max_col_num != 0
            and self.n_channels > self.max_col_num
        ):
            self.data = np.pad(
                self.data,
                ((0, 0), (0, self.max_col_num - self.n_channels % self.max_col_num)),
            )
        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        chunk_index = index // self.one_chunk_num
        data_chunk = (
            self.data[
                :, chunk_index * self.max_col_num : (chunk_index + 1) * self.max_col_num
            ]
            if (chunk_index + 1) * self.max_col_num < self.n_channels
            else self.data[:, chunk_index * self.max_col_num :]
        )
        seq_start = self.stride * (index % self.one_chunk_num)
        seq_end = seq_start + self.context_len

        pred_end = seq_end + self.horizon_len

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = pred_end - self.horizon_len
            seq_start = seq_end - self.context_len

        # input_seq = self.data[seq_start:seq_end, :].T
        input_seq = data_chunk[seq_start:seq_end, :].T
        forecast_seq = data_chunk[seq_end:pred_end, :].T
        return input_seq, forecast_seq

    def __len__(self):
        if self.length_timeseries < self.context_len + self.horizon_len:
            return 1 * self.num_chunks
        return self.num_chunks * self.one_chunk_num

    def get_data_loader(self):
        if self.mode == "train":
            # dtl = DataLoader(self, batch_size=self.batchsize, shuffle=True)
            # for i, data in enumerate(dtl):
            #     timeseries, input_mask, forecast = data
            #     print(self.data.shape)
            #     print(timeseries.shape, input_mask.shape, forecast.shape)
            #     break
            return DataLoader(self, shuffle=True, batch_size=self.batchsize)
        else:
            return DataLoader(self, shuffle=False, batch_size=self.batchsize)


class MomentDataset(BaseDataset):
    """
    Dataset class for Moment model
    Data Format:
    Dict with keys:
    input_ts: np.ndarray, historical time series data
    actual_ts: np.ndarray, actual time series data
    """

    def __init__(
        self,
        name=None,
        datetime_col=None,
        path=None,
        batchsize=64,
        mode="train",
        boundaries=[0, 0, 0],
        horizon_len=0,
        task_name="forecasting",
        label_col=None,
        stride=10,
        **kwargs,
    ):
        super().__init__(
            name=name,
            datetime_col=datetime_col,
            path=path,
            batchsize=batchsize,
            mode=mode,
        )
        self.task_name = task_name
        self.label_col = "label" if label_col is None else label_col
        self.mode = mode

        self.seq_len = 512
        self.stride = (
            stride if (self.mode == "train" or horizon_len == 0) else horizon_len
        )
        self.forecast_horizon = horizon_len
        self.boundaries = boundaries
        self.max_col_num = 64

        self.pad = False
        self._read_data()

        self.one_chunk_num = (
            self.length_timeseries - self.seq_len - self.forecast_horizon
        ) // self.stride + 1

    def _read_data(self):
        self.scaler = StandardScaler()
        self.df = pd.read_csv(self.data_path)

        if self.boundaries[0] == 0:
            self.boundaries[0] = int(len(self.df) * 0.5)
        if self.boundaries[1] == 0:
            self.boundaries[1] = int(len(self.df) * 0.7)
        if self.boundaries[2] == 0:
            self.boundaries[2] = int(len(self.df) - 1)

        if self.boundaries == [-1, -1, -1]:
            # use all data for training
            self.boundaries = [0, 0, len(self.df) - 1]

        self.forecast_horizon = min(self.forecast_horizon, int(0.3 * len(self.df) + 1))

        if self.task_name == "detection":
            self.n_channels = 1
        else:
            self.n_channels = self.df.shape[1] - 1
        self.num_chunks = (self.n_channels + self.max_col_num - 1) // self.max_col_num

        if self.datetime_col:
            self.df.drop(columns=[self.datetime_col], inplace=True)

        if self.task_name == "forecasting" or self.task_name == "imputation":
            self.df = self.df.infer_objects(copy=False).interpolate(method="cubic")
        elif self.task_name == "detection":
            self.df.interpolate(inplace=True, method="cubic")

        if self.task_name == "forecasting" or self.task_name == "imputation":
            self.scaler.fit(self.df[slice(0, int(len(self.df) * 0.5))].values)
            self.df = self.scaler.transform(self.df.values)
        elif self.task_name == "detection":
            self.labels = self.df.iloc[:, -1].values
            ts = self.df.iloc[:, 0].values.reshape(-1, 1)
            self.scaler.fit(ts[slice(0, self.boundaries[0])])
            ts = self.scaler.transform(ts)

        elif self.task_name == "classification":
            self.data, self.labels = get_multivariate_data(
                self.df, label_col=self.label_col
            )
            self.labels = self._transform_labels(self.labels)
            self.num_series, self.n_channels, self.len_timeseries = self.data.shape
            self.data = self.data.reshape(
                -1, self.len_timeseries
            )  # reshape data into (num_samples*num_channels, num_timesteps)
            self.scaler.fit(self.data)
            self.data = self.scaler.transform(self.data)

            if self.n_channels == 1:
                self.data = self.data.reshape(self.num_series, self.len_timeseries)
                self.data = self.data.T

        if self.mode == "train":
            if self.task_name == "forecasting" or self.task_name == "imputation":
                self.data = self.df[slice(0, self.boundaries[0]), :]
            elif self.task_name == "detection":
                self.data, self.labels = (
                    ts[slice(0, self.boundaries[0])],
                    self.labels[slice(0, self.boundaries[0])],
                )

        elif self.mode == "test":
            if self.task_name == "forecasting" or self.task_name == "imputation":
                self.data = self.df[slice(self.boundaries[1], self.boundaries[2]), :]
            elif self.task_name == "detection":
                self.data, self.labels = (
                    ts[slice(self.boundaries[1], self.boundaries[2])],
                    self.labels[slice(self.boundaries[1], self.boundaries[2])],
                )

        self.length_timeseries = self.data.shape[0]
        self.required_len = self.seq_len + self.forecast_horizon
        self.pad_len = 0
        if self.length_timeseries < self.required_len:
            self.pad = True
        self.pad_sequence()

    def pad_sequence(self):
        self.pad_len = self.required_len - self.length_timeseries
        # Pad data with zeros from the left
        if self.pad:
            self.data = np.pad(self.data, ((self.pad_len, 0), (0, 0)))
        # If num of channels isn't multiple of max_col_num, pad with zeros
        if self.n_channels % self.max_col_num != 0:
            self.data = np.pad(
                self.data,
                ((0, 0), (0, self.max_col_num - self.n_channels % self.max_col_num)),
            )
        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        chunk_index = index // self.one_chunk_num
        data_chunk = (
            self.data[
                :, chunk_index * self.max_col_num : (chunk_index + 1) * self.max_col_num
            ]
            if (chunk_index + 1) * self.max_col_num < self.n_channels
            else self.data[:, chunk_index * self.max_col_num :]
        )
        seq_start = self.stride * (index % self.one_chunk_num)
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)
        # if the sequence is padded, mask of padded part is 0
        input_mask[: self.pad_len] = 0

        pred_end = seq_end + self.forecast_horizon

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = pred_end - self.forecast_horizon
            seq_start = seq_end - self.seq_len

        # input_seq = self.data[seq_start:seq_end, :].T
        input_seq = data_chunk[seq_start:seq_end, :].T
        if self.task_name == "forecasting":
            # forecast_seq = self.data[seq_end:pred_end, :].T
            forecast_seq = data_chunk[seq_end:pred_end, :].T
            return input_seq, input_mask, forecast_seq
        elif self.task_name == "imputation":
            return input_seq, input_mask
        elif self.task_name == "detection":
            labels = (
                self.labels[seq_start:seq_end]
                .astype(int)
                .reshape((self.n_channels, self.seq_len))
            )
            return input_seq, input_mask, labels
        elif self.task_name == "classification":
            input_seq = self.data[:, index]
            input_seq = np.expand_dims(input_seq, axis=0)
            labels = self.labels[index,].astype(int)
            return input_seq, input_mask, labels

    def __len__(self):
        if self.task_name == "classification":
            return self.num_series
        if self.length_timeseries < self.seq_len + self.forecast_horizon:
            return 1 * self.num_chunks
        return self.num_chunks * self.one_chunk_num

    def get_data_loader(self):
        if self.mode == "train":
            # dtl = DataLoader(self, batch_size=self.batchsize, shuffle=True)
            # for i, data in enumerate(dtl):
            #     timeseries, input_mask, forecast = data
            #     print(self.data.shape)
            #     print(timeseries.shape, input_mask.shape, forecast.shape)
            #     break
            return DataLoader(self, batch_size=self.batchsize, shuffle=True)
        else:
            return DataLoader(self, batch_size=self.batchsize, shuffle=False)

    def _transform_labels(self, labels: np.ndarray):
        unq_labels = np.unique(labels)  # Move the labels to {0, ..., L-1}
        transform = {}
        for i, l in enumerate(unq_labels):
            transform[l] = i

        labels = np.vectorize(transform.get)(labels)

        return labels


class TinyTimeMixerDataset(BaseDataset):
    """
    Dataset class for ChronosBolt model
    Data Format:
    Dict with keys:
    input_ts: np.ndarray, historical time series data
    actual_ts: np.ndarray, actual time series data
    """

    def __init__(
        self,
        name=None,
        datetime_col="ds",
        path=None,
        boundaries=[0, 0, 0],
        batch_size=128,
        mode=None,
        stride=10,
        context_len=512,
        horizon_len=64,
    ):
        super().__init__(
            name=name,
            datetime_col=datetime_col,
            path=path,
            batchsize=batch_size,
            mode=mode,
        )
        # Todo: implement ChronosDataset

        self.context_len = context_len
        self.horizon_len = horizon_len
        self.mode = mode
        self.boundaries = boundaries
        self.stride = stride
        self.batchsize = batch_size
        self.max_col_num = 64

        self.pad = False
        self._read_data()

        self.one_chunk_num = (
            self.length_timeseries - self.context_len - self.horizon_len
        ) // self.stride + 1

    def _read_data(self):
        self.df = pd.read_csv(self.data_path)

        if self.boundaries[0] == 0:
            self.boundaries[0] = int(len(self.df) * 0.5)
        if self.boundaries[1] == 0:
            self.boundaries[1] = int(len(self.df) * 0.7)
        if self.boundaries[2] == 0:
            self.boundaries[2] = int(len(self.df) - 1)

        if self.boundaries == [-1, -1, -1]:
            # use all data for training
            self.boundaries = [0, 0, len(self.df) - 1]

        self.horizon_len = min(self.horizon_len, int(0.3 * len(self.df) + 1))

        self.n_channels = self.df.shape[1] - 1
        self.num_chunks = (self.n_channels + self.max_col_num - 1) // self.max_col_num

        if self.datetime_col:
            self.df.drop(columns=[self.datetime_col], inplace=True)

        self.df = np.array(self.df)

        if self.mode == "train":
            self.data = self.df[slice(0, self.boundaries[0]), :]

        elif self.mode == "test":
            self.data = self.df[slice(self.boundaries[1], self.boundaries[2]), :]

        self.length_timeseries = self.data.shape[0]
        self.required_len = self.context_len + self.horizon_len
        self.pad_len = 0
        if self.length_timeseries < self.required_len:
            self.pad = True
        self.pad_sequence()

    def pad_sequence(self):
        self.pad_len = self.required_len - self.length_timeseries
        # Pad data with zeros from the left
        if self.pad:
            self.data = np.pad(self.data, ((self.pad_len, 0), (0, 0)))
        # If num of channels isn't multiple of max_col_num, pad with zeros
        if (
            self.n_channels % self.max_col_num != 0
            and self.n_channels > self.max_col_num
        ):
            self.data = np.pad(
                self.data,
                ((0, 0), (0, self.max_col_num - self.n_channels % self.max_col_num)),
            )
        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        chunk_index = index // self.one_chunk_num
        data_chunk = (
            self.data[
                :, chunk_index * self.max_col_num : (chunk_index + 1) * self.max_col_num
            ]
            if (chunk_index + 1) * self.max_col_num < self.n_channels
            else self.data[:, chunk_index * self.max_col_num :]
        )
        seq_start = self.stride * (index % self.one_chunk_num)
        seq_end = seq_start + self.context_len

        pred_end = seq_end + self.horizon_len

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = pred_end - self.horizon_len
            seq_start = seq_end - self.context_len

        # input_seq = self.data[seq_start:seq_end, :].T
        input_seq = data_chunk[seq_start:seq_end, :].T
        forecast_seq = data_chunk[seq_end:pred_end, :].T
        return input_seq, forecast_seq

    def __len__(self):
        if self.length_timeseries < self.context_len + self.horizon_len:
            return 1 * self.num_chunks
        return self.num_chunks * self.one_chunk_num

    def get_data_loader(self):
        if self.mode == "train":
            return DataLoader(self, shuffle=True, batch_size=self.batchsize)
        else:
            return DataLoader(self, shuffle=False, batch_size=self.batchsize)
        # shape: (batch_size, n_channels, seq_len)


class LPTMDataset(BaseDataset):
    """
    Dataset class for Moment model
    Data Format:
    Dict with keys:
    input_ts: np.ndarray, historical time series data
    actual_ts: np.ndarray, actual time series data
    """

    def __init__(
        self,
        name=None,
        datetime_col=None,
        path=None,
        batchsize=16,
        mode="train",
        boundaries=[0, 0, 0],
        horizon=0,
        task_name="forecasting",
        label_col=None,
        stride=10,
        seq_len=512,
        **kwargs,
    ):
        super().__init__(
            name=name,
            datetime_col=datetime_col,
            path=path,
            batchsize=batchsize,
            mode=mode,
        )
        self.task_name = task_name
        self.label_col = "label" if label_col is None else label_col

        self.seq_len = seq_len
        self.stride = stride
        self.forecast_horizon = horizon
        self.boundaries = boundaries

        self.max_col_num = 64
        self.pad = False
        self._read_data()

        self.one_chunk_num = (
            self.length_timeseries - self.seq_len - self.forecast_horizon
        ) // self.stride + 1

    def _read_data(self):
        self.scaler = StandardScaler()
        self.df = pd.read_csv(self.data_path)

        if self.boundaries[0] == 0:
            self.boundaries[0] = int(len(self.df) * 0.6)
        if self.boundaries[1] == 0:
            self.boundaries[1] = int(len(self.df) * 0.8)
        if self.boundaries[2] == 0:
            self.boundaries[2] = int(len(self.df) - 1)

        if self.boundaries == [-1, -1, -1]:
            # use all data for training
            self.boundaries = [0, 0, len(self.df) - 1]

        self.forecast_horizon = min(self.forecast_horizon, int(0.3 * len(self.df) + 1))

        if self.task_name == "detection":
            self.n_channels = 1
        else:
            self.n_channels = self.df.shape[1] - 1

        if self.datetime_col:
            self.df.drop(columns=[self.datetime_col], inplace=True)

        if (
            self.task_name == "forecasting"
            or self.task_name == "imputation"
            or self.task_name == "forecasting2"
        ):
            self.df = self.df.infer_objects(copy=False).interpolate(method="cubic")
        elif self.task_name == "detection":
            self.df.interpolate(inplace=True, method="cubic")

        if (
            self.task_name == "forecasting"
            or self.task_name == "imputation"
            or self.task_name == "forecasting2"
        ):
            self.scaler.fit(self.df[slice(0, self.boundaries[0])].values)
            self.df = self.scaler.transform(self.df.values)
        elif self.task_name == "detection":
            self.labels = self.df.iloc[:, -1].values
            ts = self.df.iloc[:, 0].values.reshape(-1, 1)
            self.scaler.fit(ts[slice(0, self.boundaries[0])])
            ts = self.scaler.transform(ts)

        elif self.task_name == "classification":
            self.data, self.labels = get_multivariate_data(
                self.df, label_col=self.label_col
            )
            self.labels = self._transform_labels(self.labels)
            self.num_series, self.n_channels, self.len_timeseries = self.data.shape
            self.data = self.data.reshape(
                -1, self.len_timeseries
            )  # reshape data into (num_samples*num_channels, num_timesteps)
            self.scaler.fit(self.data)
            self.data = self.scaler.transform(self.data)

            if self.n_channels == 1:
                self.data = self.data.reshape(self.num_series, self.len_timeseries)
                self.data = self.data.T

        if self.mode == "train":
            if (
                self.task_name == "forecasting"
                or self.task_name == "imputation"
                or self.task_name == "forecasting2"
            ):
                self.data = self.df[slice(0, self.boundaries[0]), :]
            elif self.task_name == "detection":
                self.data, self.labels = (
                    ts[slice(0, self.boundaries[0])],
                    self.labels[slice(0, self.boundaries[0])],
                )

        elif self.mode == "test":
            if (
                self.task_name == "forecasting"
                or self.task_name == "imputation"
                or self.task_name == "forecasting2"
            ):
                self.data = self.df[slice(self.boundaries[1], self.boundaries[2]), :]
            elif self.task_name == "detection":
                self.data, self.labels = (
                    ts[slice(self.boundaries[1], self.boundaries[2])],
                    self.labels[slice(self.boundaries[1], self.boundaries[2])],
                )

        self.length_timeseries = self.data.shape[0]
        self.required_len = self.seq_len + self.forecast_horizon
        self.pad_len = 0
        if self.length_timeseries < self.required_len:
            self.pad = True
        if self.pad:
            self.pad_sequence()
        self.num_chunks = (self.n_channels + self.max_col_num - 1) // self.max_col_num

    def pad_sequence(self):
        self.pad_len = self.required_len - self.length_timeseries
        # Pad data with zeros from the left
        self.data = np.pad(self.data, ((self.pad_len, 0), (0, 0)))
        # If num of channels isn't multiple of max_col_num, pad with zeros
        if self.n_channels % self.max_col_num != 0:
            self.data = np.pad(
                self.data,
                ((0, 0), (0, self.max_col_num - self.n_channels % self.max_col_num)),
            )
        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        chunk_index = index // self.one_chunk_num
        data_chunk = (
            self.data[
                :, chunk_index * self.max_col_num : (chunk_index + 1) * self.max_col_num
            ]
            if (chunk_index + 1) * self.max_col_num < self.n_channels
            else self.data[:, chunk_index * self.max_col_num :]
        )

        seq_start = self.stride * (index % self.one_chunk_num)
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)
        # if the sequence is padded, mask of padded part is 0
        input_mask[: self.pad_len] = 0

        pred_end = seq_end + self.forecast_horizon

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = pred_end - self.forecast_horizon
            seq_start = seq_end - self.seq_len

        # input_seq = self.data[seq_start:seq_end, :].T
        input_seq = data_chunk[seq_start:seq_end, :].T
        if self.task_name == "forecasting":
            # forecast_seq = self.data[seq_end:pred_end, :].T
            forecast_seq = data_chunk[seq_end:pred_end, :].T
            return input_seq, input_mask, forecast_seq
        elif self.task_name == "imputation":
            return input_seq, input_mask
        elif self.task_name == "forecasting2":
            # input_seq = self.data[pred_end - self.seq_len : pred_end, :].T
            input_seq = data_chunk[seq_end - self.seq_len : seq_end, :].T
            input_mask[seq_end:pred_end] = 0
            input_mask[self.pad_len :] = 1
            # input_mask[: self.pad_len] = 0
            # forecast_seq = self.data[seq_end:pred_end, :].T
            forecast_seq = data_chunk[seq_end:pred_end, :].T
            return input_seq, input_mask, forecast_seq
        elif self.task_name == "detection":
            labels = (
                self.labels[seq_start:seq_end]
                .astype(int)
                .reshape((self.n_channels, self.seq_len))
            )
            return input_seq, input_mask, labels
        elif self.task_name == "classification":
            input_seq = self.data[:, index]
            input_seq = np.expand_dims(input_seq, axis=0)
            labels = self.labels[index,].astype(int)
            return input_seq, input_mask, labels

    def __len__(self):
        if self.task_name == "classification":
            return self.num_series
        if self.length_timeseries < self.seq_len + self.forecast_horizon:
            # return 1
            return 1 * self.num_chunks
        # return (
        #     self.length_timeseries - self.seq_len - self.forecast_horizon
        # ) // self.stride + 1
        return self.num_chunks * self.one_chunk_num

    def get_data_loader(self):
        if self.mode == "train":
            return DataLoader(self, batch_size=self.batchsize, shuffle=True)
        else:
            return DataLoader(self, batch_size=self.batchsize, shuffle=False)

    def _transform_labels(self, labels: np.ndarray):
        unq_labels = np.unique(labels)  # Move the labels to {0, ..., L-1}
        transform = {}
        for i, l in enumerate(unq_labels):
            transform[l] = i

        labels = np.vectorize(transform.get)(labels)

        return labels


class MoiraiDataset(BaseDataset):
    """
    Dataset class for Moirai model.
    It ingests data in the form of a (num_variates x num_timesteps) matrix.
    """

    def __init__(
        self,
        name=None,
        datetime_col="date",
        path=None,
        boundaries=(0, 0, 0),
        context_len=128,
        horizon_len=32,
        patch_size=16,
        batch_size=16,
        freq=None,
        start_date=None,
        end_date=None,
        operation="mean",
        normalize=True,
        mode="train",
        htune=False,  # hyperparameter tuning
        data_config=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            datetime_col=datetime_col,
            path=path,
            batchsize=batch_size,
            mode=mode,
        )
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.mode = mode
        self.htune = htune
        self.boundaries = boundaries
        self.normalize = normalize
        self.kwargs = kwargs
        if data_config:
            self.target_dim = data_config.get("target_dim", 1)
            self.feat_dynamic_real_dim = data_config.get("feat_dynamic_real_dim", 0)
            self.past_feat_dynamic_real_dim = data_config.get(
                "past_feat_dynamic_real_dim", 0
            )
        else:
            self.target_dim = 1
            self.feat_dynamic_real_dim = 0
            self.past_feat_dynamic_real_dim = 0

        self._read_data()  # read from path into a pandas dataframe
        # Preprocess the data - infer freq, take subset or normalize
        self._preprocess(
            start_date=start_date, end_date=end_date, freq=freq, operation=operation
        )
        self.start_date = self.dataset.index[0]
        self.train_transforms = self.default_transforms()
        self.test_transforms = self.default_transforms()

        # Split the dataset into train, val, test
        if self.mode == "train":  # no windowing
            self.dataset = self.dataset[: self.boundaries[0]]
            self.gen_train_val_data()
        elif self.mode == "val":  # no windowing
            self.dataset = self.dataset[self.boundaries[0] : self.boundaries[1]]
            self.gen_train_val_data()
        elif self.mode == "test":
            # whole dataset sent
            self.gen_test_data()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _read_data(self):
        """This function reads the data from the data_path and sets the dataset, infers frequency
        and splits the columns as index (datetime_col) and variates columns (ts_cols)
        """
        self.data = pd.read_csv(self.data_path)

        # set datetime_col as index and remove it from columns
        self.data[self.datetime_col] = pd.to_datetime(self.data[self.datetime_col])
        self.data = self.data.set_index(self.datetime_col)
        self.freq = pd.infer_freq(self.data.index)
        self.dataset = self.data
        self.ts_cols = [col for col in self.dataset.columns if col != self.datetime_col]

    def _preprocess(
        self, start_date=None, end_date=None, freq=None, operation="mean", **kwargs
    ):
        """This function picks a subset of data if start_date or end_date are provided.
        It resamples the data if freq is provided.
        It normalizes the data if normalize is set to True.
        It splits the data into train, val, test based on boundaries.

        Args:
            start_date (str, optional): Start of subset data. Defaults to None.
            end_date (str, optional): End of subset of data. Defaults to None.
            freq (str, optional): "h"(hourly), "w"(weekly), "m"(monthly), "q"(quarterly), etc for resampling. Defaults to None.
            operation (str, optional): Operation used in resampling. Defaults to 'mean'.

        Raises:
            ValueError: If operation is not supported.
        """
        # When considering a subset of the data
        if start_date:
            start_date = pd.Timestamp(start_date)
            self.dataset = self.dataset[self.dataset.index >= start_date]

        if end_date:
            end_date = pd.Timestamp(end_date)
            self.dataset = self.dataset[self.dataset.index <= end_date]

        # Fill missing values
        self.dataset = self.dataset.ffill()
        self.dataset = self.dataset.bfill()  # ensures the first row has no NaN values

        # Resample the data if required
        if freq:
            if operation == "sum":
                self.dataset = self.dataset.resample(freq).sum()
            elif operation == "mean":
                self.dataset = self.dataset.resample(freq).mean()
            elif operation == "pad":
                self.dataset = self.dataset.resample(freq).pad()
            elif operation == "ffill":
                self.dataset = self.dataset.resample(freq).ffill()
            elif operation == "bfill":
                self.dataset = self.dataset.resample(freq).bfill()
            else:
                raise ValueError(f"Unsupported resampling operation: {operation}")

        # Decide the boundaries for train, val, test
        if self.boundaries == (0, 0, 0):
            if self.htune:  # if we are doing hyperparameter tuning
                # 60% train, 20% val, 20% test
                self.boundaries = [
                    int(self.dataset.shape[0] * 0.6),
                    int(self.dataset.shape[0] * 0.8),
                    self.dataset.shape[0] - 1,
                ]
            else:
                # 80% train, 20% test
                self.boundaries = [
                    int(self.dataset.shape[0] * 0.8),
                    int(self.dataset.shape[0] * 0.8),
                    self.dataset.shape[0] - 1,
                ]

        # Normalize the dataset if required
        if self.normalize:
            print("Normalizing the dataset")
            scaler = StandardScaler()
            scaler = scaler.fit(self.dataset.iloc[: self.boundaries[1]])
            data_normalized = scaler.transform(self.dataset)
            self.dataset = pd.DataFrame(
                data_normalized, columns=self.dataset.columns, index=self.dataset.index
            )

    def gen_train_val_data(self):
        """Generates training and validation data based on the boundaries

        Returns:
            np.ndarray: Training and Validation data
        """
        data = []
        # Each column is a separate time series
        # Each time series is appended to the data list
        for i in range(self.dataset.shape[1]):
            data.append(
                {
                    "start": Period(self.start_date, freq=self.freq),
                    "target": self.dataset.iloc[:, i].values,
                    "item_id": self.dataset.columns[i],
                }
            )

        self.dataset = MoiraiTorch(data)
        self.data = data

    def gen_test_data(self):
        """Generates test data based on the boundaries

        Returns:
            np.ndarray: Test data
        """
        data = []
        num_windows = (
            1
            if (self.dataset.shape[0] - self.boundaries[1]) < self.horizon_len
            else (self.dataset.shape[0] - self.boundaries[1]) // self.horizon_len
        )
        for i in range(self.dataset.shape[1]):
            for j in range(num_windows):
                start_idx = self.boundaries[1] + j * self.horizon_len
                end_idx = start_idx + self.horizon_len
                data.append(
                    (
                        {  # input
                            "start": Period(self.start_date, freq=self.freq),
                            "target": self.dataset.iloc[:start_idx, i].values,
                            "item_id": self.dataset.columns[i],
                        },
                        {  # label
                            "start": Period(self.start_date, freq=self.freq),
                            "target": self.dataset.iloc[start_idx:end_idx, i].values,
                            "item_id": self.dataset.columns[i],
                        },
                    )
                )

        self.dataset = MoiraiTorch(data)
        self.data = data

    def default_transforms(self) -> transforms.Compose:
        """Default transformations for the dataset"""
        transforms_list = []

        # Convert the target data to numpy array
        transforms_list.append(
            AsNumpy(
                field="target",
                expected_ndim=1 if self.target_dim == 1 else 2,
                dtype=np.float32,
            )
        )

        if self.target_dim == 1:
            # Fix missing values
            transforms_list.append(
                AddObservedValues(
                    target_field="target",
                    output_field="observed_target",
                    imputation_method=CausalMeanNaNFix(),
                    dtype=bool,
                )
            )

            # Add dimension to target
            transforms_list.append(ArrExpandDims(field="target", axis=0))
            transforms_list.append(ArrExpandDims(field="observed_target", axis=0))
        else:
            transforms_list.append(
                AddObservedValues(
                    target_field="target",
                    output_field="observed_target",
                    dtype=bool,
                )
            )

        if self.feat_dynamic_real_dim > 0:
            transforms_list.append(
                AsNumpy(
                    field="feat_dynamic_real",
                    expected_ndim=2,
                    dtype=np.float32,
                )
            )
            transforms_list.append(
                AddObservedValues(
                    target_field="feat_dynamic_real",
                    output_field="observed_feat_dynamic_real",
                    dtype=bool,
                )
            )

        if self.past_feat_dynamic_real_dim > 0:
            transforms_list.append(
                AsNumpyArray(
                    field="past_feat_dynamic_real",
                    expected_ndim=2,
                    dtype=np.float32,
                )
            )
            transforms_list.append(
                AddObservedValuesIndicator(
                    target_field="past_feat_dynamic_real",
                    output_field="past_observed_feat_dynamic_real",
                    dtype=bool,
                )
            )

        # Convert list of tranforms to a single transformation
        comp_transform = transforms.Compose(transforms_list)

        return comp_transform

    @property
    def past_length(self) -> int:
        return (
            self.context_len + self.horizon_len
            if self.patch_size == "auto"
            else self.context_len
        )

    def add_past_fields(
        self,
        data: dict,
        ts_fields: list = [],
        past_ts_fields: list = [],
        dummy_val: float = 0.0,
        lead_time: int = 0,
        target_field: str = "target",
        is_pad_field: str = "is_pad",
        observed_value_field: str = "observed_target",
        start_field: str = "start",
        forecast_start_field: str = "forecast_start",
        output_NTC: bool = True,
        mode="train",
    ):
        """Add the following fields:
        (a) past_target: The past target data
        (b) past_observed_target: The past target data with missing values indicator
        (c) past_is_pad: Indicates if the added value was a padding value
        (d) past_feat_dynamic_real: The past dynamic real features
        (e) past_observed_feat_dynamic_real: The past dynamic real features with missing values indicator
        """
        pred_len = self.horizon_len
        target = data[target_field]
        num_windows = 1 + ((target.shape[-1] - self.past_length) // pred_len)

        # Sample indices from the target field using the instance sampler
        if mode == "train":
            sampled_indices = [
                self.past_length + i * pred_len for i in range(num_windows + 1)
            ]
        elif mode == "test":
            sampled_indices = custom_train_instance_split(target)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Columns to be sliced
        slice_cols = ts_fields + past_ts_fields + [target_field, observed_value_field]

        transformed_data = []
        # Iterate over the sampled indices
        for i in range(len(sampled_indices)):
            idx = sampled_indices[i]
            # Calculate the padding length if the index is less than past_length
            d = data.copy()
            pad_length = max(
                0,
                self.past_length
                - d[target_field][..., (idx - self.past_length) : idx].shape[-1],
            )

            # Iterate over the fields to be sliced
            for field in slice_cols:
                # Slice the past piece of the field
                if pad_length == 0:
                    past_piece = d[field][..., (idx - self.past_length) : idx]
                else:
                    pad_block = np.full(
                        shape=d[field].shape[:-1] + (pad_length,),
                        fill_value=dummy_val,
                        dtype=d[field].dtype,
                    )
                    past_piece = np.concatenate(
                        [pad_block, d[field][..., (idx - self.past_length) : idx]],
                        axis=-1,
                    )

                # # Slice the future piece of the field
                # future_piece = d[field][..., (idx + lead_time) : (idx + lead_time + pred_len)]
                future_piece = np.full(
                    shape=d[field].shape[:-1] + (pred_len,),
                    fill_value=dummy_val,
                    dtype=d[field].dtype,
                )

                # If the field is in time series fields, concatenate past and future pieces
                if field in ts_fields:
                    piece = np.concatenate([past_piece, future_piece], axis=-1)
                    if output_NTC:
                        piece = piece.transpose()
                    d[field] = piece
                else:
                    if output_NTC:
                        past_piece = past_piece.transpose()
                        # future_piece = future_piece.transpose()
                    if field not in past_ts_fields:
                        d["past_" + field] = past_piece
                        # d["future_" + field] = future_piece
                        del d[field]
                    else:
                        d[field] = past_piece

            # Create a padding indicator for the past piece
            pad_indicator = np.zeros(self.past_length)
            if pad_length > 0:
                pad_indicator[:pad_length] = 1
            d["past_" + (is_pad_field)] = pad_indicator

            # Set the forecast start field
            d[forecast_start_field] = (d[start_field] + idx + lead_time).to_timestamp()

            # Append the transformed data
            transformed_data.append(d)

        # Return the transformed data
        return transformed_data

    def prep_train_test_data(self, mode="train"):
        """Apply transforms on the data and add the past fields (past target, past observed target, etc)"""
        ts_fields = []
        if self.feat_dynamic_real_dim > 0:
            ts_fields.append("feat_dynamic_real")
            ts_fields.append("observed_feat_dynamic_real")
        past_ts_fields = []
        if self.past_feat_dynamic_real_dim > 0:
            past_ts_fields.append("past_feat_dynamic_real")
            past_ts_fields.append("past_observed_feat_dynamic_real")

        if mode == "train":
            # STEP 1: Apply the transforms on the data
            while self.train_transforms.transforms:
                t = self.train_transforms.transforms.pop(0)
                self.data = [t(x) for x in self.data]
            # STEP 2: Linearize the data and add the required fields
            transformed_data = []
            for x in self.data:
                transformed_data.extend(
                    self.add_past_fields(
                        data=x,
                        mode="train",
                        ts_fields=ts_fields,
                        past_ts_fields=past_ts_fields,
                    )
                )
            self.data = transformed_data
            # STEP 3: Convert the data to a MoiraiTorch object
            self.batched_data = MoiraiTorch(self.data)

        elif mode == "test":
            # STEP 1: Apply the transforms on the data
            data = [x[0] for x in self.data]  # only input part
            while self.test_transforms.transforms:
                t = self.test_transforms.transforms.pop(0)
                data = [t(x) for x in data]
            # STEP 2: Linearize the data and add the required fields
            transformed_data = []
            for x in data:
                transformed_data.extend(
                    self.add_past_fields(
                        data=x,
                        mode="test",
                        ts_fields=ts_fields,
                        past_ts_fields=past_ts_fields,
                    )
                )
            # STEP 3: Convert the data to a MoiraiTorch object
            self.batched_data = MoiraiTorch(transformed_data)

    def get_dataloader(self):
        """Returns the iterator for data batches for the dataset based on the mode

        Returns:
            torch.utils.data.DataLoader: Depends on the mode
        """
        if self.mode == "train":
            self.prep_train_test_data(mode="train")
            if self.kwargs:
                batch_size = self.kwargs.get("batch_size", self.batch_size)
                num_workers = self.kwargs.get("num_workers", 0)
                pin_memory = self.kwargs.get("pin_memory", False)
                persistent_workers = self.kwargs.get("persistent_workers", False)

                return DataLoader(
                    self.batched_data,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                )
            return DataLoader(
                self.batched_data, batch_size=self.batch_size, shuffle=True
            )
        else:
            self.prep_train_test_data(mode="test")
            return DataLoader(
                self.batched_data, batch_size=self.batch_size, shuffle=False
            )

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def __len__(self):
        return len(self.dataset[0]["target"])


class Moirai_old_Dataset(BaseDataset):
    """
    Dataset class for Moirai model
    Data Format:

    """

    def __init__(
        self,
        name=None,
        datetime_col="ds",
        path=None,
        boundaries=(0, 0, 0),
        context_len=128,
        horizon_len=32,
        patch_size="auto",
        batch_size=16,
        freq=None,
        start_date=None,
        end_date=None,
        operation="mean",
        normalize=True,
        mode="train",
        **kwargs,
    ):
        super().__init__(name=name, datetime_col=datetime_col, path=path)
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.mode = mode
        self.normalize = normalize
        self.data = pd.read_csv(self.data_path)
        # set datetime_col as index and remove it from columns
        self.data[self.datetime_col] = pd.to_datetime(self.data[self.datetime_col])
        self.data = self.data.set_index(self.datetime_col)
        self.freq = pd.infer_freq(self.data.index)
        self.dataset = self.data
        self.ts_cols = [col for col in self.dataset.columns if col != self.datetime_col]

        if start_date:
            start_date = pd.Timestamp(start_date)
            self.dataset = self.dataset[self.dataset.index >= start_date]

        if end_date:
            end_date = pd.Timestamp(end_date)
            self.dataset = self.dataset[self.dataset.index <= end_date]

        self.dataset = self.dataset.ffill()
        self.dataset = self.dataset.bfill()

        if freq:
            if operation == "sum":
                self.dataset = self.dataset.resample(freq).sum()
            elif operation == "mean":
                self.dataset = self.dataset.resample(freq).mean()
            elif operation == "pad":
                self.dataset = self.dataset.resample(freq).pad()
            elif operation == "ffill":
                self.dataset = self.dataset.resample(freq).ffill()
            elif operation == "bfill":
                self.dataset = self.dataset.resample(freq).bfill()
            else:
                raise ValueError(f"Unsupported resampling operation: {operation}")

        if boundaries == (0, 0, 0):
            # Default boundaries: train 60%, val 20%, test 20%
            self.boundaries = [
                int(len(self.data) * 0.8),
                int(len(self.data) * 0.8),
                len(self.data),
            ]
        else:
            self.boundaries = boundaries

        # Normalize the dataset if required
        if self.normalize:
            scaler = StandardScaler()
            scalar = scaler.fit(self.dataset.iloc[: self.boundaries[1]])
            data_normalized = scaler.transform(self.dataset)
            self.dataset = pd.DataFrame(
                data_normalized, columns=self.dataset.columns, index=self.dataset.index
            )

        test_offset = self.boundaries[2] - self.boundaries[1]
        self.dataset = PandasDataset(dict(self.dataset))

        train_template, test_template = ts_split(self.dataset, offset=-test_offset)

        # split the data based on boundaries
        if self.mode == "train":
            self.dataset = train_template
        elif self.mode == "val":
            self.dataset = train_template
        else:
            self.dataset = test_template.generate_instances(
                prediction_length=self.horizon_len,
                windows=test_offset // self.horizon_len,
                distance=self.horizon_len,
            )

    def get_dataloader(self):
        if self.mode == "train":
            return DataLoader(self.dataset, batch_size=self.batchsize, shuffle=True)
        elif self.mode == "val":
            return DataLoader(self.dataset, batch_size=self.batchsize, shuffle=False)
        else:
            return DataLoader(self.dataset, batch_size=self.batchsize, shuffle=False)


class TimeMoEDataset(BaseDataset):
    """
    Dataset class for TimeMoE model
    Data Format:
    Dict with keys:
    input_ts: np.ndarray, historical time series data
    actual_ts: np.ndarray, actual time series data
    """

    def __init__(
        self,
        name=None,
        datetime_col=None,
        path=None,
        batch_size=16,
        mode="train",
        boundaries=[0, 0, 0],
        task_name="evaluation",
        stride=10,
        context_len=512,
        horizon_len=96,
        **kwargs,
    ):
        super().__init__(
            name=name,
            datetime_col=datetime_col,
            path=path,
            batchsize=batch_size,
            mode=mode,
        )
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.task_name = task_name

        self.stride = stride
        self.boundaries = boundaries

        self.pad = False
        self._read_data()

    def _read_data(self):
        self.df = pd.read_csv(self.data_path)

        if self.boundaries[0] == 0:
            self.boundaries[0] = int(len(self.df) * 0.5)
        if self.boundaries[1] == 0:
            self.boundaries[1] = int(len(self.df) * 0.7)
        if self.boundaries[2] == 0:
            self.boundaries[2] = int(len(self.df) - 1)

        if self.boundaries == [-1, -1, -1]:
            # use all data for training
            self.boundaries = [0, 0, len(self.df) - 1]

        self.horizon_len = min(self.horizon_len, int(0.3 * len(self.df) + 1))

        self.n_channels = self.df.shape[1] - 1

        if self.datetime_col:
            self.df.drop(columns=[self.datetime_col], inplace=True)

        self.df = np.array(self.df)

        if self.mode == "train":
            self.data = self.df[slice(0, self.boundaries[0]), :]

        elif self.mode == "test":
            self.data = self.df[slice(self.boundaries[1], self.boundaries[2]), :]

        scaler = StandardScaler()
        scaler = scaler.fit(self.df[slice(0, self.boundaries[0]), :])
        self.data = scaler.transform(self.data)

        self.length_timeseries = self.data.shape[0]
        self.required_len = self.context_len + self.horizon_len
        self.pad_len = 0
        if self.length_timeseries < self.required_len:
            self.pad = True
        self.pad_sequence()

    def pad_sequence(self):
        self.pad_len = self.required_len - self.length_timeseries
        # Pad data with zeros from the left
        if self.pad:
            self.data = np.pad(self.data, ((self.pad_len, 0), (0, 0)))
        self.length_timeseries = self.data.shape[0]
        self.num_windows = (
            1
            + (self.length_timeseries - self.context_len - self.horizon_len)
            // self.stride
        )

    def __getitem__(self, index):
        channel_idx = index // self.num_windows
        seq_start = self.stride * (index % self.num_windows)
        seq_end = seq_start + self.context_len

        if self.task_name == "evaluation":
            pred_end = seq_end + self.horizon_len

            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = pred_end - self.horizon_len
                seq_start = seq_end - self.context_len

            # input_seq = self.data[seq_start:seq_end, :].T
            input_seq = self.data[seq_start:seq_end, channel_idx]
            forecast_seq = self.data[seq_end:pred_end, channel_idx]
            return input_seq, forecast_seq

        elif self.task_name == "finetune":
            pred_end = seq_end + 1
            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = pred_end - 1
                seq_start = seq_end - self.context_len

            input_seq = self.data[
                seq_start:seq_end, channel_idx
            ]  # shape: (context_len, )
            forecast_seq = self.data[seq_end:pred_end, channel_idx]
            loss_mask = np.ones(input_seq.shape[0])
            return input_seq, forecast_seq, loss_mask

    def __len__(self):
        if self.length_timeseries < self.context_len + self.horizon_len:
            return 1 * self.n_channels
        return self.n_channels * self.num_windows

    def get_data_loader(self):
        if self.mode == "train":
            return DataLoader(self, shuffle=True, batch_size=self.batchsize)
        else:
            return DataLoader(self, shuffle=False, batch_size=self.batchsize)
        # shape: (batch_size, n_channels, seq_len)
