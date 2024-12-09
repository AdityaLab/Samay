import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import os
from typing import List, Tuple, Union
from pathlib import Path
from gluonts.dataset.arrow import ArrowWriter
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split as ts_split


from .models.timesfm.timesfm.data_loader import TimeSeriesdata
from .models.moment.momentfm.utils.data import load_from_tsfile
from .utils import get_multivariate_data


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
class BaseDataset():
    def __init__(self, name=None, datetime_col=None, path=None, batchsize=8, mode='train', **kwargs):
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
                **kwargs):
        super().__init__(name=name, datetime_col=datetime_col, path=path, batchsize=batchsize, mode=mode)
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.freq = freq
        self.normalize = normalize
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
        if self.mode == "train":
            tfset = tfdtl.torch_dataset(mode="train", shift=1)
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
    def __init__(self, name=None, 
                 datetime_col=None, 
                 path=None, 
                 batchsize=8, 
                 mode='train', 
                 boundaries=[0, 0, 0], 
                 horizon=0, 
                 task_name='forecasting',
                 label_col=None,
                 stride=10,
                 **kwargs):
        super().__init__(name=name, datetime_col=datetime_col, path=path, batchsize=batchsize, mode=mode)
        self.task_name = task_name
        self.label_col = 'label' if label_col is None else label_col
        
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

        if self.task_name == 'detection':
            self.n_channels = 1
        else:
            self.n_channels = self.df.shape[1] - 1
        
        if self.datetime_col:
            self.df.drop(columns=[self.datetime_col], inplace=True)

        if self.task_name == 'forecasting' or self.task_name == 'imputation':
            self.df = self.df.infer_objects(copy=False).interpolate(method="cubic")
        elif self.task_name == 'detection':
            self.df.interpolate(inplace=True, method='cubic')

        if self.task_name == 'forecasting' or self.task_name == 'imputation':
            self.scaler.fit(self.df[slice(0, self.boundaries[0])].values)
            self.df = self.scaler.transform(self.df.values)
        elif self.task_name == 'detection':
            self.labels = self.df.iloc[:, -1].values
            ts = self.df.iloc[:, 0].values.reshape(-1, 1)
            self.scaler.fit(ts[slice(0, self.boundaries[0])])
            ts = self.scaler.transform(ts)

        elif self.task_name == 'classification':
            self.data, self.labels = get_multivariate_data(self.df, label_col=self.label_col)
            self.labels = self._transform_labels(self.labels)
            self.num_series, self.n_channels, self.len_timeseries = self.data.shape
            self.data = self.data.reshape(-1, self.len_timeseries) # reshape data into (num_samples*num_channels, num_timesteps)
            self.scaler.fit(self.data)
            self.data = self.scaler.transform(self.data)
            
            if self.n_channels == 1:
                self.data = self.data.reshape(self.num_series, self.len_timeseries)
                self.data = self.data.T

        if self.mode == "train":
            if self.task_name == 'forecasting' or self.task_name == 'imputation':
                self.data = self.df[slice(0, self.boundaries[0]), :]
            elif self.task_name == 'detection':
                self.data, self.labels = ts[slice(0, self.boundaries[0])], self.labels[slice(0, self.boundaries[0])]

        elif self.mode == "test":
            if self.task_name == 'forecasting' or self.task_name == 'imputation':
                self.data = self.df[slice(self.boundaries[1], self.boundaries[2]), :]
            elif self.task_name == 'detection':
                self.data, self.labels = ts[slice(self.boundaries[1], self.boundaries[2])], self.labels[slice(self.boundaries[1], self.boundaries[2])]

        self.length_timeseries = self.data.shape[0]

    def pad_sequence(self):
        self.pad_len = self.required_len - self.length_timeseries
        # Pad data with zeros from the left
        self.data = np.pad(
            self.data, ((self.pad_len, 0), (0, 0))
        )
        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        if self.pad:
            self.pad_sequence()

        seq_start = self.stride * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)
        # if the sequence is padded, mask of padded part is 0
        input_mask[:self.pad_len] = 0

        pred_end = seq_end + self.forecast_horizon

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = pred_end - self.forecast_horizon
            seq_start = seq_end - self.seq_len

        input_seq = self.data[seq_start:seq_end, :].T
        if self.task_name == 'forecasting':
            forecast_seq = self.data[seq_end:pred_end, :].T
            return input_seq, input_mask, forecast_seq
        elif self.task_name == 'imputation':
            return input_seq, input_mask
        elif self.task_name == 'detection':
            labels = (
                self.labels[seq_start:seq_end]
                .astype(int)
                .reshape((self.n_channels, self.seq_len))
            )
            return input_seq, input_mask, labels
        elif self.task_name == 'classification':
            input_seq = self.data[:, index]
            input_seq = np.expand_dims(input_seq, axis=0)
            labels = self.labels[index,].astype(int)
            return input_seq, input_mask, labels

    def __len__(self):
        if self.task_name == 'classification':
            return self.num_series
        if self.length_timeseries < self.seq_len + self.forecast_horizon:
            return 1
        return (self.length_timeseries - self.seq_len - self.forecast_horizon) // self.stride + 1
    
    def get_data_loader(self):
        if self.mode == 'train':
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