import os 
import sys
import torch
import numpy as np
import pandas as pd

src_path = os.path.abspath(os.path.join("src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from tsfmproject.model import TimesfmModel, ChronosModel, MoiraiTSModel
from tsfmproject.dataset import TimesfmDataset, ChronosDataset, MoiraiDataset
from tsfmproject.utils import load_args


DATASET_LIST = ["electricity", "exchange_rate", "illness", "traffic", "weather"]
DATASET_PATH = ["data" + "/{dataset}/{dataset}.csv".format(dataset=dataset) for dataset in DATASET_LIST] 

def update_leaderboard(dataset_name, model_name, metrics, leaderboard_path):
    """
    Updates the leaderboard file with new metrics for a given dataset and model.
    """
    if not os.path.exists(leaderboard_path):
        # Create the leaderboard with appropriate columns if it doesn't exist
        columns = ["Dataset"] + [f"{model_name}_{metric}" for model in ["TimesFM", "Chronos", "Moirai"]
                                 for metric in metrics.keys()]
        df = pd.DataFrame(columns=columns)
        df.to_csv(leaderboard_path, encoding="utf-8", index=False)
    else:
        df = pd.read_csv(leaderboard_path)
    
    # Update or add a row for the current dataset
    if dataset_name not in df["Dataset"].values:
        df = pd.concat([df, pd.DataFrame({"Dataset": [dataset_name]}, index=[len(df)])], ignore_index=True)
    
    for metric, value in metrics.items():
        column_name = f"{model_name}_{metric}"
        if column_name not in df.columns:
            df[column_name] = np.nan
        df.loc[df["Dataset"] == dataset_name, column_name] = value
    
    # Save the updated leaderboard
    df.to_csv(leaderboard_path, index=False, encoding="utf-8")
    print("Leaderboard updated and saved")
    print("Leaderboard:")


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
        torch.cuda.empty_cache()
        val_dataset = TimesfmDataset(name="ett", datetime_col='date', path=dataset_path,
                              mode='test', context_len=args["config"]["context_len"], horizon_len=args["config"]["horizon_len"], normalize=False)
        avg_loss, trues, preds, histories = tfm.evaluate(val_dataset)
        MASE = np.mean(np.abs(trues - preds))/np.mean(np.abs(trues[:,:,1:] - trues[:,:,:-1]))
        print("MSE: ", avg_loss)
        print("MASE: ", MASE)
        print("Saving to leaderboard")
        if model_name not in df.columns:
            df[model_name] = np.nan

        if dataset not in df["Dataset\Model"].values:
            df.loc[len(df), "Dataset\Model"] = dataset

        df.loc[df["Dataset\Model"] == dataset, model_name] = MAE
        df.to_csv(leaderboard_path, index=False, encoding="utf-8")
        print("Leaderboard updated and saved")
        print("Leaderboard: ")
        # print(df)

    arg_path = "config/chronos.json"
    model_name = "Chronos"
    args = load_args(arg_path)
    chronos = ChronosModel(config=args["config"], repo=args["repo"])
    chronos.load_model()

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
        torch.cuda.empty_cache()
        val_dataset = ChronosDataset(name="ett", datetime_col='date', path=dataset_path,
                              mode='test', context_len=args["config"]["context_len"], horizon_len=args["config"]["horizon_len"], normalize=False)
        eval_results, trues, preds, histories = chronos.evaluate(val_dataset, batch_size=8, metrics=["MSE", "MASE"])
        print("MSE: ", eval_results["MSE"])
        print("MASE: ", eval_results["MASE"])
        print("Saving to leaderboard")
        if model_name not in df.columns:
            df[model_name] = np.nan

        if dataset not in df["Dataset\Model"].values:
            df.loc[len(df), "Dataset\Model"] = dataset

        df.loc[df["Dataset\Model"] == dataset, model_name] = MAE
        df.to_csv(leaderboard_path, index=False, encoding="utf-8")
        print("Leaderboard updated and saved")
        print("Leaderboard: ")
        # print(df)

    arg_path = "config/moirai.json"
    model_name = "Moirai"
    args = load_args(arg_path)
    moirai = MoiraiTSModel(config=args["config"], repo=args["repo"], model_type=args["model_type"], model_size=args["model_size"])

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
        torch.cuda.empty_cache()
        val_dataset = MoiraiDataset(name="ett", datetime_col='date', path=dataset_path,
                              mode='test', context_len=args["config"]["context_len"], horizon_len=args["config"]["horizon_len"], normalize=False)
        eval_results, trues, preds, histories = moirai.evaluate(val_dataset, metrics=["MSE", "MASE"])
        print("MSE: ", eval_results["MSE"])
        print("MASE: ", eval_results["MASE"])
        print("Saving to leaderboard")
        if model_name not in df.columns:
            df[model_name] = np.nan

        if dataset not in df["Dataset\Model"].values:
            df.loc[len(df), "Dataset\Model"] = dataset

        df.loc[df["Dataset\Model"] == dataset, model_name] = MAE
        df.to_csv(leaderboard_path, index=False, encoding="utf-8")
        print("Leaderboard updated and saved")
        print("Leaderboard: ")
        # print(df)