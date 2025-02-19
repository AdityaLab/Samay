import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

from .models.timesfm.timesfm.data_loader import TimeSeriesdata
from .models.moment.momentfm.utils.data import load_from_tsfile
from .models.chronosforecasting.chronos.chronos import MeanScaleUniformBins, ChronosConfig
from .utils import get_multivariate_data


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
                batchsize=4,
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
            # Default boundaries: train 50%, val 20%, test 30%
            self.boundaries = [
                int(len(self.data) * 0.5),
                int(len(self.data) * 0.7),
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
            batch_size=16,
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
        super().__init__(name=name, datetime_col=datetime_col, path=path, batchsize=batch_size, mode=mode)
        # Todo: implement ChronosDataset
        assert tokenizer_class is not None, "Tokenizer is required for ChronosDataset"
        
        if not config:
            self.config = ChronosConfig(
                tokenizer_class="MeanScaleUniformBins",
                tokenizer_kwargs={'low_limit': -15.0, 'high_limit': 15.0},
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
        assert type(self.config) == ChronosConfig, "Config must be an instance of ChronosConfig"
        assert self.config.model_type in ("seq2seq", "causal"), "Model type must be either 'seq2seq' or 'causal'"

        if tokenizer_class == "MeanScaleUniformBins":
            self.tokenizer = MeanScaleUniformBins(**self.config.tokenizer_kwargs, config=self.config)
        else:
            raise ValueError(f"Tokenizer class {tokenizer_class} not supported")
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
        self.max_col_num = 64

        self.pad = False
        self._read_data()
        self.preprocess()

        self.one_chunk_num = (self.length_timeseries - self.context_len - self.horizon_len) // self.stride + 1

    def _read_data(self):
        self.df = pd.read_csv(self.data_path)

        if self.boundaries[0] == 0:
            self.boundaries[0] = int(len(self.df) * 0.5)
        if self.boundaries[1] == 0:
            self.boundaries[1] = int(len(self.df) * 0.7)
        if self.boundaries[2] == 0:
            self.boundaries[2] = int(len(self.df) - 1)

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
            self.data = np.pad(
                self.data, ((self.pad_len, 0), (0, 0))
            )
        # If num of channels isn't multiple of max_col_num, pad with zeros
        if self.n_channels % self.max_col_num != 0:
            self.data = np.pad(
                self.data, ((0, 0), (0, self.max_col_num - self.n_channels % self.max_col_num))
            )
        self.length_timeseries = self.data.shape[0]


    def __getitem__(self, index):
        chunk_index = index // self.one_chunk_num
        data_chunk = self.data[:, chunk_index * self.max_col_num: (chunk_index + 1) * self.max_col_num] if (chunk_index + 1) * self.max_col_num < self.n_channels else self.data[:, chunk_index * self.max_col_num:]
        seq_start = self.stride * index
        seq_end = seq_start + self.context_len
        input_mask = np.ones(self.context_len)
        # if the sequence is padded, mask of padded part is 0
        input_mask[:self.pad_len] = 0

        pred_end = seq_end + self.horizon_len

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = pred_end - self.horizon_len
            seq_start = seq_end - self.context_len

        # input_seq = self.data[seq_start:seq_end, :].T
        input_seq = data_chunk[seq_start:seq_end, :].T
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(torch.tensor(input_seq))
        forecast_seq = data_chunk[seq_end:pred_end, :].T
        labels, labels_mask = self.tokenizer.label_input_transform(torch.tensor(forecast_seq), scale)
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
        if self.mode == 'train':
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
                 batchsize=64, 
                 mode='train', 
                 boundaries=[0, 0, 0], 
                 horizon_len=0, 
                 task_name='forecasting',
                 label_col=None,
                 stride=10,
                 **kwargs):
        super().__init__(name=name, datetime_col=datetime_col, path=path, batchsize=batchsize, mode=mode)
        self.task_name = task_name
        self.label_col = 'label' if label_col is None else label_col
        self.mode = mode
        
        self.seq_len = 512
        self.stride = stride if self.mode == 'train' else horizon_len
        self.forecast_horizon = horizon_len
        self.boundaries = boundaries
        self.max_col_num = 64

        self.pad = False
        self._read_data()
        
        self.one_chunk_num = (self.length_timeseries - self.seq_len - self.forecast_horizon) // self.stride + 1

    def _read_data(self):
        self.scaler = StandardScaler()
        self.df = pd.read_csv(self.data_path)

        if self.boundaries[0] == 0:
            self.boundaries[0] = int(len(self.df) * 0.5)
        if self.boundaries[1] == 0:
            self.boundaries[1] = int(len(self.df) * 0.7)
        if self.boundaries[2] == 0:
            self.boundaries[2] = int(len(self.df) - 1)

        if self.task_name == 'detection':
            self.n_channels = 1
        else:
            self.n_channels = self.df.shape[1] - 1
        self.num_chunks = (self.n_channels + self.max_col_num - 1) // self.max_col_num
        
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
        self.required_len = self.seq_len + self.forecast_horizon
        self.pad_len = 0
        if self.length_timeseries < self.required_len:
            self.pad = True
        self.pad_sequence()

    def pad_sequence(self):
        self.pad_len = self.required_len - self.length_timeseries
        # Pad data with zeros from the left
        if self.pad:
            self.data = np.pad(
                self.data, ((self.pad_len, 0), (0, 0))
            )
        # If num of channels isn't multiple of max_col_num, pad with zeros
        if self.n_channels % self.max_col_num != 0:
            self.data = np.pad(
                self.data, ((0, 0), (0, self.max_col_num - self.n_channels % self.max_col_num))
            )
        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        chunk_index = index // self.one_chunk_num
        data_chunk = self.data[:, chunk_index * self.max_col_num: (chunk_index + 1) * self.max_col_num] if (chunk_index + 1) * self.max_col_num < self.n_channels else self.data[:, chunk_index * self.max_col_num:]
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

        # input_seq = self.data[seq_start:seq_end, :].T
        input_seq = data_chunk[seq_start:seq_end, :].T
        if self.task_name == 'forecasting':
            # forecast_seq = self.data[seq_end:pred_end, :].T
            forecast_seq = data_chunk[seq_end:pred_end, :].T
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
            return 1 * self.num_chunks
        return self.num_chunks * self.one_chunk_num
    
    def get_data_loader(self):
        if self.mode == 'train':
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


if __name__ == "__main__":
    from .models.chronosforecasting.chronos.chronos import MeanScaleUniformBins, ChronosConfig

    chronos_config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs="{'low_limit': -15.0, 'high_limit': 15.0}",
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
    tokenizer = MeanScaleUniformBins(low_limit=-15.0, high_limit=15.0, config=chronos_config)
    dataset = ChronosDataset(name="ett", datetime_col='date', path='tsfmproject/models/moment/data/ETTh1.csv',
                              mode='train', context_len=512, horizon_len=64, tokenizer=tokenizer, model_type="seq2seq")
    print(len(dataset))
    print(dataset[0])
        
