import os
import sys

import numpy as np
import pandas as pd

src_path = os.path.abspath(os.path.join("src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from samay.dataset import TimesfmDataset
from samay.model import TimesfmModel
from samay.utils import load_args

DATASET_LIST = ["electricity", "exchange_rate", "illness", "traffic", "weather"]
DATASET_PATH = [
    "data" + "/{dataset}/{dataset}.csv".format(dataset=dataset)
    for dataset in DATASET_LIST
]


if __name__ == "__main__":
    # Load the model
    arg_path = "config/timesfm.json"
    model_name = "TimesFM"
    args = load_args(arg_path)
    tfm = TimesfmModel(**args)

    for dataset, dataset_path in zip(DATASET_LIST, DATASET_PATH):
        print("Creating leaderboard for dataset: ", dataset)
        # The Leaderboard has column header for different model names, each row is a different dataset, first column is the dataset name
        # The Leaderboard is saved as a csv file
        # Create the Leaderboard if it does not exist
        leaderboard_dir = os.path.join("leaderboard")
        if not os.path.exists(leaderboard_dir):
            os.makedirs(leaderboard_dir)
        leaderboard_path = os.path.join(leaderboard_dir, "leaderboard.csv")
        if not os.path.exists(leaderboard_path):
            df = pd.DataFrame(columns=["Dataset\Model"])
            df.to_csv(leaderboard_path, encoding="utf-8", index=False)
        df = pd.read_csv(leaderboard_path)
        val_dataset = TimesfmDataset(
            name="ett",
            datetime_col="date",
            path=dataset_path,
            mode="test",
            context_len=args["config"]["context_len"],
            horizon_len=args["config"]["horizon_len"],
            normalize=False,
        )
        avg_loss, trues, preds, histories = tfm.evaluate(val_dataset)
        MAE = np.mean(np.abs(trues - preds))
        print("MAE: ", MAE)
        print("Saving to leaderboard")
        if model_name not in df.columns:
            df[model_name] = np.nan

        if dataset not in df["Dataset\Model"].values:
            df.loc[len(df), "Dataset\Model"] = dataset

        df.loc[df["Dataset\Model"] == dataset, model_name] = MAE
        df.to_csv(leaderboard_path, index=False, encoding="utf-8")
        print("Leaderboard updated and saved")
        print("Leaderboard: ")
        print(df)
