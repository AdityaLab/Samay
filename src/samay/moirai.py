import os
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from pandas._libs.tslibs.period import Period
from datasets import load_dataset
from gluonts.dataset.arrow import ArrowWriter
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split as ts_split
import gluonts.dataset.loader as glu_load
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .models.timesfm.timesfm.data_loader import TimeSeriesdata
from .utils import get_multivariate_data


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

        return context, actual


class MoiraiTorch(Dataset):
    def __init__(self,data:list[dict]):
        super().__init__()
        self.data = data
        self.input = [d["input"] for d in data if "input" in d]
        self.label = [d["label"] for d in data if "label" in d]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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

        self._read_data()
        self._preprocess(start_date=start_date, end_date=end_date,
                        freq=freq, operation=operation)
        self.start_date = self.dataset.index[0]
        
        # Split the dataset into train, val, test
        if self.mode == "train": # no windowing
            self.dataset = self.dataset[:self.boundaries[0]]
            self.gen_train_val_data()
        elif self.mode == "val": # no windowing
            self.dataset = self.dataset[self.boundaries[0]:self.boundaries[1]]
            self.gen_train_val_data()
        elif self.mode == "test":
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
    
    def gen_test_data(self):
        """Generates test data based on the boundaries

        Returns:
            np.ndarray: Test data
        """
        # Steps to generate test data
        # 1) Split them into windows - each of length horizon_len
        # 2) For each window, split them into input and label
        data = []
        for i in range(self.dataset.shape[1]):
            data.extend([{"input": {"start":Period(self.start_date, freq=self.freq),
                                    "target":self.dataset[:self.boundaries[1] + j*self.horizon_len].values,
                                    "item_id": self.dataset.columns[i]},
                           "label": {"start":Period(self.start_date, freq=self.freq),
                                     "target":self.dataset[self.boundaries[1] + j*self.horizon_len:self.boundaries[1] + (j+1)*self.horizon_len].values,
                                     "item_id": self.dataset.columns[i]}
                        } for j in range((self.dataset.shape[0] - self.boundaries[1])//self.horizon_len)])
        
        self.dataset = MoiraiTorch(data)
    
    def get_data_loader(self):
        """Returns the iterator for data batches for the dataset based on the mode

        Returns:
            torch.utils.data.DataLoader: Depends on the mode
        """
        if self.mode == "train":
            return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        else:
            return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)
    
    def __len__(self):
        return len(self.dataset[0]["target"])