import os
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from gluonts.dataset.arrow import ArrowWriter
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .models.timesfm.timesfm.data_loader import TimeSeriesdata
from .utils import get_multivariate_data
from .moirai_utils import (MoiraiTorch,
    AsNumpy,
    AddObservedValues,
    ArrExpandDims,
    CausalMeanNaNFix,
    custom_train_instance_split
)
from pandas._libs.tslibs.period import Period

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split as ts_split
from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader, InferenceDataLoader
from gluonts.transform import (
    AddObservedValuesIndicator,
    AsNumpyArray,
    CausalMeanValueImputation,
    ExpandDimArray,
    TestSplitSampler,
    Transformation,
)

from torchvision import transforms

# for full length history/ context data wrapping
# class TimeSeriesDataset(Dataset):
#     """
#     A PyTorch Dataset for sliding window extraction from time series data.
#     """
#     def __init__(self, data, boundary1, boundary2, context_len, horizon_len, stride=-1):
#         """
#         Initialize the dataset with sliding window logic.

#         Args:
#             data (pd.DataFrame): The input time series data.
#             context_len (int): Length of the context window.
#             horizon_len (int): Length of the forecast horizon.
#             stride (int): Step size for sliding the window.
#         """
#         self.data = data
#         self.context_len = context_len
#         self.horizon_len = horizon_len
#         self.total_len = context_len + horizon_len
#         self.stride = stride

#         if(self.stride == -1):
#             self.stride = self.horizon_len

#         # Generate start indices for sliding windows
#         self.indices = [
#             start
#             for start in range(boundary1, boundary2 - self.total_len + 1, self.stride)
#         ]

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         start = self.indices[idx]
#         window = self.data.iloc[ : start + self.total_len]

#         # Extract context and actuals, and convert to Torch tensors
#         context = torch.tensor(window.iloc[: -self.horizon_len].to_numpy().transpose(), dtype=torch.float32)
#         actual = torch.tensor(window.iloc[-self.horizon_len :].to_numpy().transpose(), dtype=torch.float32)

#         # # Return the input as a list of tensors (one for each column)
#         # input_list = [context[i] for i in range(context.shape[0])]

#         return context, actual

# for fixed length history/ context data wrapping
class TimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset for sliding window extraction from time series data.
    """
    def __init__(self, data, context_len, horizon_len, stride=-1):
        """
        Initialize the dataset with sliding window logic.

        Args:
            data (pd.DataFrame): The input time series data.
            context_len (int): Length of the context window.
            horizon_len (int): Length of the forecast horizon.
            stride (int): Step size for sliding the window.
        """
        self.data = data
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.total_len = context_len + horizon_len
        self.stride = stride

        if(self.stride == -1):
            self.stride = self.horizon_len

        # Generate start indices for sliding windows
        self.indices = [
            start
            for start in range(0, len(data) - self.total_len + 1, self.stride)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        window = self.data.iloc[start : start + self.total_len]

        # Extract context and actuals, and convert to Torch tensors
        context = torch.tensor(window.iloc[: self.context_len].to_numpy().transpose(), dtype=torch.float32)
        actual = torch.tensor(window.iloc[self.context_len :].to_numpy().transpose(), dtype=torch.float32)
        start_date = window.index[0]

        # # Return the input as a list of tensors (one for each column)
        # input_list = [context[i] for i in range(context.shape[0])]

        return context, actual


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
    def __init__(self, name=None,
                datetime_col='ds',
                path=None,
                batchsize=16,
                mode='train',
                boundaries=(0, 0, 0),
                context_len=128,
                horizon_len=32, 
                freq='h', 
                normalize=True, 
                stride=10,
                **kwargs):
        super().__init__(name=name, datetime_col=datetime_col, path=path, batchsize=batchsize, mode=mode)
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.freq = freq
        self.normalize = normalize
        self.stride = stride
        self.data = pd.read_csv(self.data_path)
        if boundaries == (0, 0, 0):
            # Default boundaries: train 60%, val 20%, test 20%
            self.boundaries = [
                int(len(self.data) * 0.6),
                int(len(self.data) * 0.8),
                len(self.data) - 1,
            ]
        else:
            self.boundaries = boundaries
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
            batch_size=self.batchsize,
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
        past_ts = data[0].reshape(self.batchsize * len(self.ts_cols), -1)
        actual_ts = data[3].reshape(self.batchsize * len(self.ts_cols), -1)
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
    Tuple of 2 elements:
    input/context: np.ndarray, historical time series data
    actual: np.ndarray, actual time series data
    """

    def __init__(
        self,
        name=None,
        datetime_col="ds",
        path=None,
        boundaries=(0, 0, 0),
        context_len=128,
        horizon_len=32,
        batch_size=16,
        freq = None,
        start_date=None,
        end_date=None,
        operation='sum',
        normalize=True,
        mode="train",
        **kwargs,
    ):
        super().__init__(name=name, datetime_col=datetime_col, path=path)
        self.context_len = context_len
        self.horizon_len = horizon_len
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
            if operation == 'sum':
                self.dataset = self.dataset.resample(freq).sum()
            elif operation == 'mean':
                self.dataset = self.dataset.resample(freq).mean()
            elif operation == 'pad':
                self.dataset = self.dataset.resample(freq).pad()
            elif operation == 'ffill':
                self.dataset = self.dataset.resample(freq).ffill()
            elif operation == 'bfill':
                self.dataset = self.dataset.resample(freq).bfill()
            else:
                raise ValueError(f"Unsupported resampling operation: {operation}")

        if boundaries == (0, 0, 0):
            # Default boundaries: train 60%, val 20%, test 20%
            self.boundaries = [
                int(len(self.data)*0.8),
                int(len(self.data)*0.8),
                len(self.data) - 1,
            ]
        else:
            self.boundaries = boundaries

        # Normalize the dataset if required
        if self.normalize:
            scaler = StandardScaler()
            scalar = scaler.fit(self.dataset.iloc[: self.boundaries[1]])
            data_normalized = scaler.transform(self.dataset)
            self.dataset = pd.DataFrame(data_normalized, columns=self.dataset.columns, index=self.dataset.index)

        
        # split the data based on boundaries 
        if self.mode == "train":
            self.dataset = self.dataset.iloc[: self.boundaries[0]]
        elif self.mode == "val":
            self.dataset = self.dataset.iloc[self.boundaries[0] : self.boundaries[1]]
        else:
            self.dataset = self.dataset.iloc[self.boundaries[1] : ]
            self.dataset = TimeSeriesDataset(self.dataset, self.context_len, self.horizon_len)
        

    def get_data_loader(self):
        if self.mode == "test":
            return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def preprocess(self, start_date=None, end_date=None, freq=None, operation='sum', **kwargs):
        """
        Preprocess the dataset by clipping based on start_date and end_date,
        and resampling the data based on frequency change.

        Args:
            start_date (str): The start date to clip the dataset.
            end_date (str): The end date to clip the dataset.
            freq (str): The frequency to resample the dataset.
        """
        if start_date:
            start_date = pd.Timestamp(start_date)
            self.dataset = self.dataset[self.dataset.index >= start_date]
        
        if end_date:
            end_date = pd.Timestamp(end_date)
            self.dataset = self.dataset[self.dataset.index <= end_date]
        
        if freq:
            if operation == 'sum':
                self.dataset = self.dataset.resample(freq).sum()
            elif operation == 'mean':
                self.dataset = self.dataset.resample(freq).mean()
            elif operation == 'pad':
                self.dataset = self.dataset.resample(freq).pad()
            elif operation == 'ffill':
                self.dataset = self.dataset.resample(freq).ffill()
            elif operation == 'bfill':
                self.dataset = self.dataset.resample(freq).bfill()
            else:
                raise ValueError(f"Unsupported resampling operation: {operation}")

        # Normalize the dataset if required
        if self.normalize:
            self.dataset = (self.dataset - self.dataset.mean()) / self.dataset.std()
        
    def convert_to_arrow(
        self,
        path: Union[str, Path],
        time_series: Union[List[np.ndarray], np.ndarray],
        start_date: str = None,
        start_date_list: List[str] = None,
        freq: str = "D",
        compression: str = "lz4",
    ):
        assert isinstance(time_series, list) or (
            isinstance(time_series, np.ndarray) and time_series.ndim == 2
        )
        if start_date is None and start_date_list is None:
            raise ValueError("Either start_date or start_date_list must be provided.")
        if start_date_list is not None:
            dataset = [
                {"start": start_date_list[i], "target": ts, "freq": freq} for i, ts in enumerate(time_series)
            ]
        else:
            dataset = [
                {"start": start_date, "target": ts, "freq": freq} for ts in time_series
            ]
        # create the directory and files mentioned in path, if not present
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ArrowWriter(compression=compression).write_to_file(
            dataset,
            path=path,
        )


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
        batchsize=8,
        mode="train",
        boundaries=[0, 0, 0],
        horizon=0,
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

        self.seq_len = 512
        self.stride = stride
        self.forecast_horizon = horizon
        self.boundaries = boundaries

        self._read_data()
        self.required_len = self.seq_len + self.forecast_horizon
        self.pad = False
        self.pad_len = 0
        if self.length_timeseries < self.required_len:
            self.pad = True

    def _read_data(self):
        self.scaler = StandardScaler()
        self.df = pd.read_csv(self.data_path)

        if self.boundaries[0] == 0:
            self.boundaries[0] = int(len(self.df) * 0.6)
        if self.boundaries[1] == 0:
            self.boundaries[1] = int(len(self.df) * 0.8)
        if self.boundaries[2] == 0:
            self.boundaries[2] = int(len(self.df) - 1)

        if self.task_name == "detection":
            self.n_channels = 1
        else:
            self.n_channels = self.df.shape[1] - 1

        if self.datetime_col:
            self.df.drop(columns=[self.datetime_col], inplace=True)

        if self.task_name == "forecasting" or self.task_name == "imputation":
            self.df = self.df.infer_objects(copy=False).interpolate(method="cubic")
        elif self.task_name == "detection":
            self.df.interpolate(inplace=True, method="cubic")

        if self.task_name == "forecasting" or self.task_name == "imputation":
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

    def pad_sequence(self):
        self.pad_len = self.required_len - self.length_timeseries
        # Pad data with zeros from the left
        self.data = np.pad(self.data, ((self.pad_len, 0), (0, 0)))
        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        if self.pad:
            self.pad_sequence()

        seq_start = self.stride * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)
        # if the sequence is padded, mask of padded part is 0
        input_mask[: self.pad_len] = 0

        pred_end = seq_end + self.forecast_horizon

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = pred_end - self.forecast_horizon
            seq_start = seq_end - self.seq_len

        input_seq = self.data[seq_start:seq_end, :].T
        if self.task_name == "forecasting":
            forecast_seq = self.data[seq_end:pred_end, :].T
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
            return 1
        return (
            self.length_timeseries - self.seq_len - self.forecast_horizon
        ) // self.stride + 1

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
        patch_size="auto",
        batch_size=16,
        freq = None,
        start_date=None,
        end_date=None,
        operation='mean',
        normalize=True,
        mode="train",
        htune=False, # hyperparameter tuning
        data_config=None,
        **kwargs,
    ):
        super().__init__(name=name, datetime_col=datetime_col, path=path, batchsize=batch_size, mode=mode)
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
            self.past_feat_dynamic_real_dim = data_config.get("past_feat_dynamic_real_dim", 0)
        else:
            self.target_dim = 1
            self.feat_dynamic_real_dim = 0
            self.past_feat_dynamic_real_dim = 0

        self._read_data() # read from path into a pandas dataframe
        # Preprocess the data - infer freq, take subset or normalize
        self._preprocess(start_date=start_date, end_date=end_date,
                        freq=freq, operation=operation)
        self.start_date = self.dataset.index[0]
        self.train_transforms = self.default_transforms()
        
        # Split the dataset into train, val, test
        if self.mode == "train": # no windowing
            self.dataset = self.dataset[:self.boundaries[0]]
            self.gen_train_val_data()
        elif self.mode == "val": # no windowing
            self.dataset = self.dataset[self.boundaries[0]:self.boundaries[1]]
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
    
    def _preprocess(self,start_date=None, end_date=None,
                    freq=None, operation='mean',**kwargs):
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
        self.dataset = self.dataset.bfill() # ensures the first row has no NaN values

        # Resample the data if required
        if freq:
            if operation == 'sum':
                self.dataset = self.dataset.resample(freq).sum()
            elif operation == 'mean':
                self.dataset = self.dataset.resample(freq).mean()
            elif operation == 'pad':
                self.dataset = self.dataset.resample(freq).pad()
            elif operation == 'ffill':
                self.dataset = self.dataset.resample(freq).ffill()
            elif operation == 'bfill':
                self.dataset = self.dataset.resample(freq).bfill()
            else:
                raise ValueError(f"Unsupported resampling operation: {operation}")

        # Decide the boundaries for train, val, test
        if self.boundaries == (0,0,0):
            if self.htune: # if we are doing hyperparameter tuning
                # 60% train, 20% val, 20% test
                self.boundaries = [int(self.dataset.shape[0]*0.6),
                                   int(self.dataset.shape[0]*0.8),
                                   self.dataset.shape[0]-1]
            else:
                # 80% train, 20% test
                self.boundaries = [int(self.dataset.shape[0]*0.8),
                                   int(self.dataset.shape[0]*0.8),
                                   self.dataset.shape[0]-1]

        # Normalize the dataset if required
        if self.normalize:
            print("Normalizing the dataset")
            scaler = StandardScaler()
            scaler = scaler.fit(self.dataset.iloc[: self.boundaries[1]])
            data_normalized = scaler.transform(self.dataset)
            self.dataset = pd.DataFrame(data_normalized, columns=self.dataset.columns, index=self.dataset.index)
    
    def gen_train_val_data(self):
        """Generates training and validation data based on the boundaries

        Returns:
            np.ndarray: Training and Validation data
        """
        data = []
        for i in range(self.dataset.shape[1]):
            data.append({
                "start": Period(self.start_date, freq=self.freq),
                "target": self.dataset.iloc[:,i].values,
                "item_id": self.dataset.columns[i]
            })
        
        self.dataset = MoiraiTorch(data)
        self.data = data
    
    def gen_test_data(self):
        """Generates test data based on the boundaries

        Returns:
            np.ndarray: Test data
        """
        data = []
        for i in range(self.dataset.shape[1]):
            data.extend([tuple([{"start":Period(self.start_date, freq=self.freq),
                                    "target":self.dataset.iloc[:self.boundaries[1] + j*self.horizon_len,i].values,
                                    "item_id": self.dataset.columns[i]},
                                {"start":Period(self.start_date, freq=self.freq),
                                     "target":self.dataset.iloc[self.boundaries[1] + j*self.horizon_len:self.boundaries[1] + (j+1)*self.horizon_len,i].values,
                                     "item_id": self.dataset.columns[i]}
                                ]) for j in range((self.dataset.shape[0] - self.boundaries[1])//self.horizon_len)
                        ])
        
        self.dataset = MoiraiTorch(data)
        self.data = data
    
    def default_transforms(self) -> transforms.Compose:
        """Default transformations for the dataset
        """
        transforms_list = []

        # Convert the target data to numpy array
        transforms_list.append(AsNumpy(
            field="target",
            expected_ndim=1 if self.target_dim == 1 else 2,
            dtype=np.float32,
        ))

        if self.target_dim == 1:
            # Fix missing values
            transforms_list.append(AddObservedValues(
                target_field="target",
                output_field="observed_target",
                imputation_method=CausalMeanNaNFix(),
                dtype=bool,
            ))

            # Add dimension to target
            transforms_list.append(ArrExpandDims(field="target", axis=0))
            transforms_list.append(ArrExpandDims(field="observed_target", axis=0))
        else:
            transforms_list.append(AddObservedValues(
                target_field="target",
                output_field="observed_target",
                dtype=bool,
            ))

        if self.feat_dynamic_real_dim > 0:
            transforms_list.append(AsNumpy(
                field="feat_dynamic_real",
                expected_ndim=2,
                dtype=np.float32,
            ))
            transforms_list.append(AddObservedValues(
                target_field="feat_dynamic_real",
                output_field="observed_feat_dynamic_real",
                dtype=bool,
            ))

        if self.past_feat_dynamic_real_dim > 0:
            transforms_list.append(AsNumpyArray(
                field="past_feat_dynamic_real",
                expected_ndim=2,
                dtype=np.float32,
            ))
            transforms_list.append(AddObservedValuesIndicator(
                target_field="past_feat_dynamic_real",
                output_field="past_observed_feat_dynamic_real",
                dtype=bool,
            ))
        
        # Convert list of tranforms to a single transformation
        comp_transform = transforms.Compose(transforms_list)
        
        return comp_transform
    
    @property
    def past_length(self) -> int:
        return self.context_len + self.horizon_len if self.patch_size == "auto" else self.context_len
    
    def add_past_fields(self, data: dict, ts_fields:list=[],
                        past_ts_fields:list=[],dummy_val:float=0.0,
                        lead_time: int = 0, target_field: str = "target",
                        is_pad_field: str = "is_pad", observed_value_field: str = "observed_target",
                        start_field: str = "start", forecast_start_field: str = "forecast_start",
                        output_NTC: bool = True):
        """Add the following fields:
        (a) past_target: The past target data
        (b) past_observed_target: The past target data with missing values indicator
        (c) past_is_pad: Indicates if the added value was a padding value
        (d) past_feat_dynamic_real: The past dynamic real features
        (e) past_observed_feat_dynamic_real: The past dynamic real features with missing values indicator
        """
        pred_len = self.horizon_len
        target = data[target_field]
        observed_field = data[observed_value_field]
        num_windows = 1 + ((target.shape[-1] - self.past_length) // pred_len)

        # Sample indices from the target field using the instance sampler
        # sampled_indices = custom_train_instance_split(target) - to be modified later
        sampled_indices = [self.past_length + i*pred_len for i in range(num_windows+1)]

        # Columns to be sliced
        slice_cols = ts_fields + past_ts_fields + [target_field, observed_value_field]


        transformed_data = []
        # Iterate over the sampled indices
        for i in range(len(sampled_indices)):
            idx = sampled_indices[i]
            # Calculate the padding length if the index is less than past_length
            d = data.copy()
            pad_length = max(0, self.past_length - d[target_field][...,(idx - self.past_length) : idx].shape[-1])

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
                        [pad_block, d[field][...,(idx - self.past_length) : idx]], axis=-1
                    )
                
                # # Slice the future piece of the field
                # future_piece = d[field][..., (idx + lead_time) : (idx + lead_time + pred_len)]
                future_piece = np.full(shape=d[field].shape[:-1] + (pred_len,),
                                        fill_value=dummy_val,
                                        dtype=d[field].dtype)
                
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
            if isinstance(d[start_field], pd.Timestamp):
                d[forecast_start_field] = (d[start_field] + idx + lead_time).timestamp()

            # Append the transformed data
            transformed_data.append(d)

        # Return the transformed data
        return transformed_data

    def convert_for_moirai_format(self):
        """Given dataset having the following fields:
        (a) past_target: The past target data
        (b) past_observed_target: The past target data with missing values indicator
        (c) past_is_pad: Indicates if the added value was a padding value
        (d) past_feat_dynamic_real: The past dynamic real features
        (e) past_observed_feat_dynamic_real: The past dynamic real features with missing values indicator
        (f) future_target: The future target data
        (g) future_observed_target: The future target data with missing values indicator
        (h) future_is_pad: Indicates if the added value was a padding value
        (i) future_feat_dynamic_real: The future dynamic real features
        (j) future_observed_feat_dynamic_real: The future dynamic real features with missing values indicator
        (k) time: The time index
        
        Convert the data to have the following fields:
        (a) target: Batched time series data
        (b) observed_mask: Binary mask for the context part
        (c) prediction_mask: Binary mask for the prediction part
        (d) time_id: Time index
        (e) variate_id: Variate index
        """

        # Refer _convert function in forecast.py
        pass


    def prep_train_data(self):
        """Convert the input `data` to have the following fields:
        +--------------------+--------------------------------------+-----------------------+----------------------------------+
        | FIELD              | DESCRIPTION                          | TYPE                  | SHAPE                            |
        +--------------------+--------------------------------------+-----------------------+----------------------------------+
        | target             | Batched time series data             | torch.tensor[float]   | (batch_size, seq_len, max_patch) |
        | observed_mask      | Binary mask for the context part     | torch.tensor[bool]    | (batch_size, seq_len, max_patch) |
        | prediction_mask    | Binary mask for the prediction part  | torch.tensor[bool]    | (batch_size, seq_len)            |
        | time_id            | Time index                           | torch.tensor[int]     | (batch_size, seq_len)            |
        +--------------------+--------------------------------------+-----------------------+----------------------------------+
        """

        # Steps
        # (a) Apply the transforms on the data
        while self.train_transforms.transforms:
            t = self.train_transforms.transforms.pop(0)
            self.data = [t(x) for x in self.data]
        # (b) Linearize the data and add the required fields
        transformed_data = []
        for x in self.data:
            transformed_data.extend(self.add_past_fields(x))
        self.data = transformed_data
        # (c) Call _convert function to add fields like observed mask, etc.
        # (d) Convert the data to a MoiraiTorch object
        self.batched_data = MoiraiTorch(self.data)

        # data_iter = iter(self.dataset)
        
        # # Add the required fields
        # mod_data = []
        # for x in data_iter:
        #     mod_data.append({
        #         "start": x["start"],
        #         "target": x["target"],
        #         "observed_mask": torch.tensor([1] * (len(x["target"]) - self.horizon_len) + [0] * self.horizon_len, dtype=torch.bool),
        #         "prediction_mask": torch.tensor([0] * (len(x["target"]) - self.horizon_len) + [1] * self.horizon_len, dtype=torch.bool),
        #         "time_id": torch.tensor(list(range(len(x["target"]))), dtype=torch.int32),
        #         "variate_id": torch.tensor([list(self.data.columns).index(x["item_id"])]*len(x["target"]), dtype=torch.int32),
        #         "patch_size": torch.tensor([16]*len(x["target"]), dtype=torch.int32),      
        #     })

        # # construct the rolling windows
        # # rolling_data = []
        # # n = (mod_data[0]["target"].shape[0] - self.context_len) // self.horizon_len

        # # for i in range(n):
        # #     data = []
        # #     for x in mod_data:
        # #         data.append({
        # #             "start": x["start"],
        # #             "target": x["target"][:(i+1)*self.horizon_len + self.context_len],
        # #             "observed_mask": [1]*(self.context_len + i*self.horizon_len) + [0]*self.horizon_len,
        # #             "prediction_mask": [0]*(self.context_len + i*self.horizon_len) + [1]*self.horizon_len,
        # #             "time_id": x["time_id"][:(i+1)*self.horizon_len + self.context_len],
        # #             "variate_id": x["variate_id"][:(i+1)*self.horizon_len + self.context_len],
        # #             "patch_size": x["patch_size"][:(i+1)*self.horizon_len + self.context_len]
        # #         })
        # #     rolling_data.append(data)

        # rolling_data = []

        # # for rolling evaluation
        # n = mod_data[0]["target"].shape[0] - self.context_len - self.horizon_len

        # for i in range(0, n, self.horizon_len):
        #     data = []
        #     for x in mod_data:
        #         data.append({
        #             "start": x["start"],
        #             "target": x["target"][i:i + self.horizon_len + self.context_len],
        #             "observed_mask": [1]*(self.context_len) + [0]*self.horizon_len,
        #             "prediction_mask": [0]*(self.context_len) + [1]*self.horizon_len,
        #             "time_id": x["time_id"][i:i + self.horizon_len + self.context_len],
        #             "variate_id": x["variate_id"][i:i + self.horizon_len + self.context_len],
        #             "patch_size": x["patch_size"][i:i + self.horizon_len + self.context_len]
        #         })
        #     rolling_data.append(data)

        
        # # Convert the multivariate to univariate as per MOIRAI
        # batched_data = []
        # for p in rolling_data:
        #     batched_data.append({
        #         "start": torch.tensor([x["start"] for x in p], dtype=torch.float32),
        #         "target": torch.tensor([x for y in p for x in y["target"]], dtype=torch.float32),
        #         "observed_mask": torch.tensor([x for y in p for x in y["observed_mask"]], dtype=torch.bool),
        #         "prediction_mask": torch.tensor([x for y in p for x in y["prediction_mask"]], dtype=torch.bool),
        #         "time_id": torch.tensor([x for y in p for x in y["time_id"]], dtype=torch.int64),
        #         "variate_id": torch.tensor([x for y in p for x in y["variate_id"]], dtype=torch.int64),
        #         "patch_size": torch.tensor([x for y in p for x in y["patch_size"]], dtype=torch.int64)
        #         })
        
        # for x in batched_data:
        #     x["sample_id"] = torch.tensor(list(range(self.batch_size))*(x["target"].shape[0]//self.batch_size), dtype=torch.int32)
        #     # unsqueeze(-1) adds a new dimension at the end
        #     # we reshape it to (patch_size, -1)
        #     x["target"] = x["target"].unsqueeze(-1).reshape(self.patch_size, -1)
        #     x["observed_mask"] = x["observed_mask"].unsqueeze(-1).reshape(self.patch_size, -1)

        # self.batched_data = MoiraiTorch(batched_data)
    
    # def prep_test_data(self, data):
    #     """give just the time series data for testing
    #     """
    #     self.dataset = self.dataset[self.boundaries[1]:]
    #     self.gen_test_data()
    
    def get_dataloader(self):
        """Returns the iterator for data batches for the dataset based on the mode

        Returns:
            torch.utils.data.DataLoader: Depends on the mode
        """
        if self.mode == "train":
            self.prep_train_data()
            if self.kwargs:
                batch_size = self.kwargs.get("batch_size", self.batch_size)
                num_workers = self.kwargs.get("num_workers", 0)
                pin_memory = self.kwargs.get("pin_memory", False)
                persistent_workers = self.kwargs.get("persistent_workers", False)

                return DataLoader(self.batched_data, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
            return DataLoader(self.batched_data, batch_size=self.batch_size, shuffle=True)
        else:
            return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
    
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
        freq = None,
        start_date=None,
        end_date=None,
        operation='mean',
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
            if operation == 'sum':
                self.dataset = self.dataset.resample(freq).sum()
            elif operation == 'mean':
                self.dataset = self.dataset.resample(freq).mean()
            elif operation == 'pad':
                self.dataset = self.dataset.resample(freq).pad()
            elif operation == 'ffill':
                self.dataset = self.dataset.resample(freq).ffill()
            elif operation == 'bfill':
                self.dataset = self.dataset.resample(freq).bfill()
            else:
                raise ValueError(f"Unsupported resampling operation: {operation}")

        if boundaries == (0, 0, 0):
            # Default boundaries: train 60%, val 20%, test 20%
            self.boundaries = [
                int(len(self.data)*0.8),
                int(len(self.data)*0.8),
                len(self.data),
            ]
        else:
            self.boundaries = boundaries

        # Normalize the dataset if required
        if self.normalize:
            scaler = StandardScaler()
            scalar = scaler.fit(self.dataset.iloc[: self.boundaries[1]])
            data_normalized = scaler.transform(self.dataset)
            self.dataset = pd.DataFrame(data_normalized, columns=self.dataset.columns, index=self.dataset.index)

        test_offset = self.boundaries[2] - self.boundaries[1]
        self.dataset = PandasDataset(dict(self.dataset))

        train_template, test_template = ts_split(self.dataset, offset=-test_offset)

        
        # split the data based on boundaries 
        if self.mode == "train":
            self.dataset = train_template
        elif self.mode == "val":
            self.dataset = train_template
        else:
            self.dataset = test_template.generate_instances(prediction_length=self.horizon_len, windows=test_offset//self.horizon_len, distance=self.horizon_len)
    
    def get_dataloader(self):
        if self.mode == "train":
            return DataLoader(self.dataset, batch_size=self.batchsize, shuffle=True)
        elif self.mode == "val":
            return DataLoader(self.dataset, batch_size=self.batchsize, shuffle=False)
        else:
            return DataLoader(self.dataset, batch_size=self.batchsize, shuffle=False)