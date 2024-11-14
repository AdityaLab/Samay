import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
from typing import List, Tuple, Union
from pathlib import Path
from gluonts.dataset.arrow import ArrowWriter


from .models.timesfm.timesfm.data_loader import TimeSeriesdata


# function for specific dataset to download and preprocess data, returning path
# BaseDataset class call the specific function decided by "name" argument
class BaseDataset:
    def __init__(self, name=None, datetime_col="ds", path=None, **kwargs):
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

    def __init__(
        self,
        name=None,
        datetime_col="ds",
        path=None,
        boundaries=(0, 0, 0),
        context_len=128,
        horizon_len=32,
        batch_size=16,
        freq="h",
        normalize=True,
        mode="train",
        **kwargs,
    ):
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
            batch_size=self.batch_size,
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

    def __init__(
        self,
        name=None,
        datetime_col="ds",
        path=None,
        boundaries=(0, 0, 0),
        context_len=128,
        horizon_len=32,
        batch_size=16,
        freq="h",
        normalize=True,
        mode="train",
        **kwargs,
    ):
        super().__init__(name=name, datetime_col=datetime_col, path=path)
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.batch_size = batch_size
        self.freq = freq
        self.normalize = normalize
        self.data = pd.read_csv(self.data_path)
        self.dataset = self.data
        self.mode = mode

    def process_covid_data(self, start_date="2020-06-01", end_date="2021-07-31", freq='D'):
        us_columns = [col for col in self.data.columns if col.startswith('UNITED STATES')]
        df_us = self.data[['ds'] + us_columns]

        df_us.loc[:, 'ds'] = pd.to_datetime(df_us['ds'])

        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        df_us = df_us[(df_us['ds'] >= start_date) & (df_us['ds'] <= end_date)]

        df_us.set_index('ds', inplace=True)
        df_us = df_us.resample(freq).sum().reset_index()

        processed_df = pd.DataFrame(columns=['unique_id', 'startdate', 'enddate', 'infected', 'dead'])

        for col in us_columns:
            if col.endswith('_Unspecified'):
                region = col.replace('_Unspecified', '')
                infected_series = df_us[['ds', col]].copy()
                infected_series.columns = ['enddate', 'infected']
                infected_series['unique_id'] = region
                infected_series['startdate'] = df_us['ds'].min()

                dead_col = region + '_Dead'
                if dead_col in df_us.columns:
                    infected_series['dead'] = df_us[dead_col].values

                if infected_series['infected'].sum() > 0:
                    infected_series['dead'] = infected_series['dead'].clip(lower=0)
                    processed_df = pd.concat([processed_df, infected_series], ignore_index=True)
        
        infected_df = processed_df.pivot(index='enddate', columns='unique_id', values='infected')
        dead_df = processed_df.pivot(index='enddate', columns='unique_id', values='dead')

        infected_df.columns = [f"{col}_infected" for col in infected_df.columns]
        dead_df.columns = [f"{col}_dead" for col in dead_df.columns]

        self.dataset = pd.concat([infected_df, dead_df], axis=1)

        

    def pivot_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        infected_df = df.pivot(index='enddate', columns='unique_id', values='infected')
        dead_df = df.pivot(index='enddate', columns='unique_id', values='dead')

        infected_df.columns = [f"{col}_infected" for col in infected_df.columns]
        dead_df.columns = [f"{col}_dead" for col in dead_df.columns]

        pivoted_df = pd.concat([infected_df, dead_df], axis=1)

        return pivoted_df

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

        ArrowWriter(compression=compression).write_to_file(
            dataset,
            path=path,
        )
