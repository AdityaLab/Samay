import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from timesfm.src.timesfm.data_loader import TimeSeriesdata
import numpy as np
from datasets import load_dataset


# function for specific dataset to download and preprocess data, returning path
# BaseDataset class call the specific function decided by "name" argument
class BaseDataset():
    def __init__(self, name=None, datetime_col='ds', path=None, **kwargs):
        """
        Args:
            name: str, dataset name
            target: np.ndarray, target data
        """
        self.name = name
        self.datetime_col = datetime_col
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


class TimesfmDataset(BaseDataset):
    """
    Dataset class for TimesFM model
    Data Format:
    Dict with keys:
    input_ts: np.ndarray, historical time series data
    actual_ts: np.ndarray, actual time series data
    """
    def __init__(self, name=None, datetime_col='ds', path=None, boundaries=(0, 0, 0), context_len=128, horizon_len=32, batch_size=16, freq='h', normalize=True, mode='train', **kwargs):
        super().__init__(name=name, datetime_col=datetime_col, path=path)
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.batch_size = batch_size
        self.freq = freq
        self.normalize = normalize
        self.data = pd.read_csv(self.data_path)
        self.mode = mode
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
        tfdtl = TimeSeriesdata(data_path=self.data_path,
            datetime_col=self.datetime_col,
            num_cov_cols=None,
            cat_cov_cols=None,
            ts_cols=np.array(self.ts_cols),
            train_range=[0, self.boundaries[0]],
            val_range=[self.boundaries[0], self.boundaries[1]],
            test_range=[self.boundaries[1], self.boundaries[2]],
            hist_len=self.context_len,
            pred_len=self.horizon_len,
            batch_size=self.batch_size,
            freq=self.freq,
            normalize=self.normalize,
            epoch_len=None,
            holiday=False,
            permute=False,
        )
        if self.mode == 'train':
            tfset = tfdtl.torch_dataset(mode='train', shift=1)
        else:
            tfset = tfdtl.torch_dataset(mode='test', shift=self.horizon_len)
        self.dataset = tfset

    def get_data_loader(self):
        if self.mode == 'train':
            return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        else:
            return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        
    def preprocess_train_batch(self, data):
        past_ts = data[0].reshape(self.batch_size * len(self.ts_cols), -1)
        actual_ts = data[3].reshape(self.batch_size * len(self.ts_cols), -1)
        return {"input_ts": past_ts, "actual_ts": actual_ts}
    
    def preprocess_eval_batch(self, data):
        past_ts = data[0]
        actual_ts = data[3]
        return {"input_ts": past_ts, "actual_ts": actual_ts}

    def preprocess(self, data):
        pass



class ChronosDataset(BaseDataset):
    """
    Dataset class for Chronos model
    Data Format:
    Dict with keys:
    input_ts: np.ndarray, historical time series data
    actual_ts: np.ndarray, actual time series data
    """
    def __init__(self, name=None, datetime_col='ds', boundaries=(0, 0, 0), context_len=128, horizon_len=32, batch_size=16, freq='H', normalize=True, mode=None, **kwargs):
        super().__init__(name=name, datetime_col=datetime_col)
        # Todo: implement ChronosDataset
        pass

    def preprocess(self, data):
        # Todo: implement preprocess
        pass