import os 
import sys
import torch
import numpy as np
import pandas as pd

src_path = os.path.abspath(os.path.join("src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from samay.model import TimesfmModel, MomentModel
from samay.dataset import TimesfmDataset, MomentDataset
from samay.utils import load_args


MONASH_NAMES = [
    # "weather",
    "tourism_yearly",
    "tourism_quarterly",
    "tourism_monthly",
    "cif_2016",
    # "london_smart_meters",
    "australian_electricity_demand",
    # "wind_farms_minutely",
    "bitcoin",
    "pedestrian_counts",
    "vehicle_trips",
    "kdd_cup_2018",
    "nn5_daily",
    "nn5_weekly",
    # "kaggle_web_traffic",
    # "kaggle_web_traffic_weekly",
    "solar_10_minutes",
    "solar_weekly",
    # "car_parts",
    "fred_md",
    "traffic_hourly",
    "traffic_weekly",
    "hospital",
    "covid_deaths",
    "sunspot",
    "saugeenday",
    "us_births",
    "solar_4_seconds",
    "wind_4_seconds",
    # "rideshare",
    "oikolab_weather",
    "temperature_rain"
]
PRED_LEN = 	[
            # 30, 
             4, 
             8, 
             24, 
             12, 
            #  60, 
             60, 
            #  60, 
             30, 
             48, 
             30, 
             48, 
             56, 
             8, 
            #  59, 
            #  8, 
             60, 
             5, 
            #  12, 
             12,  
             48, 
             8, 
             12, 
             30, 
             30, 
             30, 
             30, 
             60, 
             60, 
            #  48, 
             48, 
             30
             ]

MONASH_PATH = ["data" + "/monash/{dataset}/test/data.csv".format(dataset=dataset) for dataset in MONASH_NAMES]
DATASET_LIST = MONASH_NAMES
DATASET_PATH = MONASH_PATH

if __name__ == "__main__":
    # Load the model
    arg_path = "config/timesfm.json"
    model_name = "TimesFM"
    args = load_args(arg_path)

    for dataset, dataset_path, pred_len in zip(DATASET_LIST, DATASET_PATH, PRED_LEN):
        args["config"]["horizon_len"] = pred_len
        tfm = TimesfmModel(**args)
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
        val_dataset = TimesfmDataset(name="ett", datetime_col='timestamp', path=dataset_path,
                              mode='test', context_len=args["config"]["context_len"], horizon_len=args["config"]["horizon_len"], normalize=False)
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







    
