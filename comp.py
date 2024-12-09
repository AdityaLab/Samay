import os
import sys
import torch
import numpy as np
import pandas as pd
import gc

src_path = os.path.abspath(os.path.join("src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from tsfmproject.model import TimesfmModel, ChronosModel, MoiraiTSModel
from tsfmproject.dataset import TimesfmDataset, ChronosDataset, MoiraiDataset
from tsfmproject.utils import load_args


DATASET_LIST = ["exchange_rate", "illness", "weather"]
DATASET_PATH = ["data" + "/{dataset}/{dataset}.csv".format(dataset=dataset) for dataset in DATASET_LIST]


def update_leaderboard(dataset_name, model_name, metrics, leaderboard_path):
    """
    Updates the leaderboard file with new metrics for a given dataset and model.
    """
    if not os.path.exists(leaderboard_path):
        # Create the leaderboard with appropriate columns if it doesn't exist
        columns = ["Dataset"] + [f"{model}_{metric}" for model in ["TimesFM", "Chronos", "Moirai"] 
                                 for metric in metrics.keys()]
        df = pd.DataFrame(columns=columns)
        df.to_csv(leaderboard_path, encoding="utf-8", index=False)
    else:
        df = pd.read_csv(leaderboard_path)
    
    # Ensure unique column names and reset index
    if not df.columns.is_unique:
        print("Duplicate columns detected. Fixing...")
        df = df.loc[:, ~df.columns.duplicated()]
    df = df.reset_index(drop=True)

    # Update or add a row for the current dataset
    if dataset_name not in df["Dataset"].values:
        new_row = pd.DataFrame({"Dataset": [dataset_name]})
        df = pd.concat([df, new_row], ignore_index=True)

    # Add metrics to the appropriate columns
    for metric, value in metrics.items():
        column_name = f"{model_name}_{metric}"
        if column_name not in df.columns:
            df[column_name] = np.nan
        df.loc[df["Dataset"] == dataset_name, column_name] = value

    # Save the updated leaderboard
    df.to_csv(leaderboard_path, index=False, encoding="utf-8")
    print("Leaderboard updated and saved")
    print("Leaderboard:")
    # print(df)


if __name__ == "__main__":
    leaderboard_dir = os.path.join("leaderboard")
    os.makedirs(leaderboard_dir, exist_ok=True)
    leaderboard_path = os.path.join(leaderboard_dir, "leaderboard1.csv")

    # # Evaluate TimesFM model
    # arg_path = "config/timesfm.json"
    # model_name = "TimesFM"
    # args = load_args(arg_path)
    # tfm = TimesfmModel(**args)

    # for dataset, dataset_path in zip(DATASET_LIST, DATASET_PATH):
    #     print(f"Evaluating {model_name} on dataset: {dataset}")
    #     torch.cuda.empty_cache()
    #     val_dataset = TimesfmDataset(name=dataset, datetime_col="date", path=dataset_path,
    #                                  mode="test", context_len=args["config"]["context_len"],
    #                                  horizon_len=args["config"]["horizon_len"], normalize=False)
    #     avg_loss, trues, preds, _ = tfm.evaluate(val_dataset)
    #     mase = np.mean(np.abs(trues - preds)) / np.mean(np.abs(trues[:, :, 1:] - trues[:, :, :-1]))
    #     metrics = {"MSE": avg_loss, "MASE": mase}
    #     update_leaderboard(dataset, model_name, metrics, leaderboard_path)

    # Evaluate Chronos model
    # arg_path = "config/chronos.json"
    # model_name = "Chronos"
    # args = load_args(arg_path)
    # chronos = ChronosModel(config=args["config"], repo=args["repo"])
    # chronos.load_model()

    # for dataset, dataset_path in zip(DATASET_LIST, DATASET_PATH):
    #     print(f"Evaluating {model_name} on dataset: {dataset}")
    #     torch.cuda.empty_cache()
    #     val_dataset = ChronosDataset(name=dataset, datetime_col="date", path=dataset_path,
    #                                  mode="test", context_len=args["config"]["context_len"],
    #                                  horizon_len=args["config"]["horizon_len"], normalize=False)
    #     eval_results, _, _, _ = chronos.evaluate(val_dataset, batch_size=8, metrics=["MSE", "MASE"])
    #     metrics = {"MSE": eval_results["MSE"], "MASE": eval_results["MASE"]}
    #     update_leaderboard(dataset, model_name, metrics, leaderboard_path)
        
    # del chronos
    # torch.cuda.empty_cache()
    # gc.collect()


    # Evaluate Moirai model
    arg_path = "config/moirai.json"
    model_name = "Moirai"
    args = load_args(arg_path)
    moirai = MoiraiTSModel(config=args["config"], repo=args["repo"], model_type=args["config"]["model_type"], model_size=args["config"]["model_size"])

    for dataset, dataset_path in zip(DATASET_LIST, DATASET_PATH):
        print(f"Evaluating {model_name} on dataset: {dataset}")
        torch.cuda.empty_cache()
        val_dataset = MoiraiDataset(name=dataset, datetime_col="date", path=dataset_path,
                                    mode="test", context_len=args["config"]["context_len"],
                                    horizon_len=args["config"]["horizon_len"], normalize=False)
        eval_results, _, _, _ = moirai.evaluate(val_dataset, metrics=["MSE", "MASE"])
        metrics = {"MSE": eval_results["MSE"], "MASE": eval_results["MASE"]}
        update_leaderboard(dataset, model_name, metrics, leaderboard_path)
    del chronos
    torch.cuda.empty_cache()
    gc.collect()