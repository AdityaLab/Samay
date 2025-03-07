import os
import datasets
from typing import Iterable, Iterator
from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.itertools import Map
from gluonts.transform import Transformation
from pathlib import Path
from toolz import compose

import pandas as pd


PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}

def itemize_start(data_entry: DataEntry) -> DataEntry:
    data_entry["start"] = data_entry["start"].item()
    return data_entry

class MultivariateToUnivariate(Transformation):
    def __init__(self, field):
        self.field = field

    def __call__(
        self, data_it: Iterable[DataEntry], is_train: bool = False
    ) -> Iterator:
        for data_entry in data_it:
            item_id = data_entry["item_id"]
            val_ls = list(data_entry[self.field])
            for id, val in enumerate(val_ls):
                data_entry[self.field] = val
                data_entry["item_id"] = item_id + "_dim" + str(id)
                yield data_entry

if __name__ == "__main__":
    gift_eval_path = Path("data/gifteval")
    dataset_names = []
    for dataset_dir in gift_eval_path.iterdir():
        if dataset_dir.name.startswith("."):
            continue
        if dataset_dir.is_dir():
            freq_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
            if freq_dirs:
                for freq_dir in freq_dirs:
                    dataset_names.append(f"{dataset_dir.name}/{freq_dir.name}")
            else:
                dataset_names.append(dataset_dir.name)

    for dataset_name in dataset_names:
        storage_path = gift_eval_path / dataset_name
        print(f"Processing {storage_path}")
        hf_dataset = datasets.load_from_disk(str(storage_path)).with_format(
            "numpy"
        )
        freq = hf_dataset[0]["freq"]
        target_dim = target.shape[0] if len((target := hf_dataset[0]["target"]).shape) > 1 else 1
        process = ProcessDataEntry(
            freq,
            one_dim_target=target_dim == 1,
        )
        gluonts_dataset = Map(compose(process, itemize_start), hf_dataset)
        
        # if it's multivariate, convert it to univariate
        if target_dim > 1:
            gluonts_dataset = MultivariateToUnivariate("target").apply(
                gluonts_dataset
            )

        timestamps = None
        series_data = {}
        for series_id, entry in enumerate(gluonts_dataset):
            start = entry["start"]
            start = start.to_timestamp()
            target = entry["target"]

            current_timestamps = pd.date_range(start=start, periods=len(target), freq=freq)

            if timestamps is None:
                timestamps = current_timestamps
            else:
                timestamps = timestamps.union(current_timestamps)

            series_data[f"series_{series_id}"] = pd.Series(target, index=current_timestamps)
            # fill in missing values in target with previous value        

        df = pd.DataFrame(series_data, index=timestamps).reset_index()
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)

        df.rename(columns={"index": "timestamp"}, inplace=True)

        df.to_csv(storage_path / "data.csv", index=False)
        print(f"Finished processing {storage_path}")

        # break
