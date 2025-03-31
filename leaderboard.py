import os 
import sys
import numpy as np
import pandas as pd
import time

# src_path = os.path.abspath(os.path.join("src"))
# if src_path not in sys.path:
#     sys.path.insert(0, src_path)

from src.samay.model import TimesfmModel, MomentModel, ChronosModel, ChronosBoltModel, TinyTimeMixerModel, MoiraiTSModel
from src.samay.dataset import TimesfmDataset, MomentDataset, ChronosDataset, ChronosBoltDataset, TinyTimeMixerDataset, MoiraiDataset
from src.samay.utils import load_args, get_gifteval_datasets
from src.samay.metric import *


# ECON_NAMES = {
#     "m4_yearly": ["Y"],
#     "m4_quarterly": ["Q"],
#     "m4_monthly": ["M"],
#     "m4_weekly": ["W"],
#     "m4_daily": ["D"],
#     "m4_hourly": ["H"],
# }

# SALES_NAMES = {
#     "car_parts_with_missing": ['M'],
#     "hierarchical_sales": ['D', 'W'],
#     "restaurant": ['D'],
# }

start = time.time()
NAMES, filesizes = get_gifteval_datasets("data/gifteval")
end = time.time()

print(f"Time taken to load datasets: {end-start:.2f} seconds")

MODEL_NAMES = ["moirai", "chronos", "chronosbolt", "timesfm", "moment", "ttm"]
MODEL_CONTEXT_LEN = {
    "timesfm": 32,
    "moment": 512,
    "chronos": 512
}


def calc_pred_and_context_len(freq):
    # split feq into base and multiplier
    base = freq[-1]
    mult = int(freq[:-1]) if len(freq) > 1 else 1
    if base == 'Y':
        pred_len = 4
        context_len = MODEL_CONTEXT_LEN["timesfm"]
    elif base == 'Q':
        pred_len = 4
        context_len = MODEL_CONTEXT_LEN["timesfm"]
    elif base == 'M':
        pred_len = 12 // mult
        context_len = MODEL_CONTEXT_LEN["timesfm"]
    elif base == 'W':
        pred_len = 4 // mult
        context_len = MODEL_CONTEXT_LEN["timesfm"]
    elif base == 'D':
        pred_len = 7 // mult
        context_len = MODEL_CONTEXT_LEN["timesfm"]
    elif base == 'H':
        pred_len = 24 // mult
        context_len = 2 * MODEL_CONTEXT_LEN["timesfm"]
    elif base == 'S':
        pred_len = 60 // mult
        context_len = 4 * MODEL_CONTEXT_LEN["timesfm"]
    else:
        raise ValueError(f"Invalid frequency: {freq}")
    return pred_len, context_len
    
    

if __name__ == "__main__":
    
    for model_name in MODEL_NAMES[4:]:
        print(f"Evaluating model: {model_name}")
        # create csv file for leaderboard if not already created
        csv_path = f"leaderboard/{model_name}.csv"
        if not os.path.exists(csv_path):
            print(f"Creating leaderboard csv file: {csv_path}")
            df = pd.DataFrame(columns=["dataset", "size_in_MB", "eval_time", "mse", "mae", "mase", "mape", "rmse", "nrmse", "smape", "msis", "nd", "mwsq", "crps"])
            df.to_csv(csv_path, index=False)

        # Load model config
        if model_name == "timesfm":
            arg_path = "config/timesfm.json"
            args = load_args(arg_path)
        elif model_name == "moment":
            arg_path = "config/moment_forecast.json"
            args = load_args(arg_path)
        elif model_name == "chronos":
            arg_path = "config/chronos.json"
            args = load_args(arg_path)
        elif model_name == "ttm":
            arg_path = "config/tinytimemixer.json"
            args = load_args(arg_path)
        elif model_name == "moirai":
            arg_path = "config/moirai.json"
            args = load_args(arg_path)

        for fname, freq, fs in filesizes:
            print(f"Evaluating {fname} ({freq})")
            # Adjust the context and prediction length based on the frequency
            # pred_len, context_len = calc_pred_and_context_len(freq)
            pred_len, context_len = 96, 512
            if model_name == "timesfm":
                args["config"]["horizon_len"] = pred_len
                args["config"]["context_len"] = context_len
            elif model_name == "moment":
                args["config"]["forecast_horizon"] = pred_len
            elif model_name == "ttm":
                args["config"]["horizon_len"] = pred_len
                args["config"]["context_len"] = context_len
            elif model_name == "moirai":
                args["config"]["horizon_len"] = pred_len
                args["config"]["context_len"] = context_len
            
            # Set the dataset path
            if len(NAMES.get(fname)) == 1:
                dataset_path = f"data/gifteval/{fname}/data.csv"
            else:
                dataset_path = f"data/gifteval/{fname}/{freq}/data.csv"
            
            # Initialize the model and dataset
            if model_name == "timesfm":
                model = TimesfmModel(**args)
                dataset = TimesfmDataset(datetime_col='timestamp', path=dataset_path, mode='test', context_len=args["config"]["context_len"], horizon_len=args["config"]["horizon_len"], boundaries=(-1, -1, -1), batchsize=64)
                start = time.time()
                metrics = model.evaluate(dataset)
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(f"Time taken for evaluation of {fname}: {end-start:.2f} seconds")

            elif model_name == "moment":
                model = MomentModel(**args)
                args["config"]["task_name"] = "forecasting"
                train_dataset = MomentDataset(datetime_col='timestamp', path=dataset_path, mode='train', horizon_len=args["config"]["forecast_horizon"], normalize=False)
                dataset = MomentDataset(datetime_col='timestamp', path=dataset_path, mode='test', horizon_len=args["config"]["forecast_horizon"], normalize=False)
                finetuned_model = model.finetune(train_dataset, task_name="forecasting")
                start = time.time()
                metrics = model.evaluate(dataset, task_name="forecasting")
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(f"Time taken for evaluation of {fname}: {end-start:.2f} seconds")
                print(metrics)

            elif model_name == "chronos":
                model = ChronosModel(**args)
                dataset_config = load_args("config/chronos_dataset.json")
                dataset_config["context_length"] = context_len
                dataset_config["prediction_length"] = pred_len
                dataset = ChronosDataset(datetime_col='timestamp', path=dataset_path, mode='test', config=dataset_config, batch_size=4)
                start = time.time()
                metrics = model.evaluate(dataset, horizon_len=dataset_config["prediction_length"], quantile_levels=[0.1, 0.5, 0.9])
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(f"Time taken for evaluation of {fname}: {end-start:.2f} seconds")

            elif model_name == "chronosbolt":
                repo = "amazon/chronos-bolt-small"
                model = ChronosBoltModel(repo=repo)
                dataset = ChronosBoltDataset(datetime_col='timestamp', path=dataset_path, mode='test', batch_size=8, context_len=context_len, horizon_len=pred_len)
                start = time.time()
                metrics = model.evaluate(dataset, horizon_len=pred_len, quantile_levels=[0.1, 0.5, 0.9])
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(f"Time taken for evaluation of {fname}: {end-start:.2f} seconds")

            elif model_name == "ttm":
                model = TinyTimeMixerModel(**args)
                dataset = TinyTimeMixerDataset(datetime_col='timestamp', path=dataset_path, mode='test', context_len=context_len, horizon_len=pred_len)
                start = time.time()
                metrics = model.evaluate(dataset)
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(f"Time taken for evaluation of {fname}: {end-start:.2f} seconds")
            
            elif model_name == "moirai":
                model = MoiraiTSModel(**args)
                dataset = MoiraiDataset(name=fname,datetime_col='timestamp', freq=freq,
                                        path=dataset_path, mode='test', context_len=context_len, horizon_len=pred_len)

                start = time.time()
                metrics = model.evaluate(dataset,leaderboard=True)
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(f"Time taken for evaluation of {fname}: {end-start:.2f} seconds")
            
            print("Evaluation done!")

            eval_time = end - start
            unit = "s"
            if eval_time > 1000: # convert to minutes
                eval_time = eval_time / 60
                unit = "m"


            df = pd.read_csv(csv_path)
            if fname in df["dataset"].values:
                df.loc[df["dataset"] == fname, "size_in_MB"] = round(fs,2)
                df.loc[df["dataset"] == fname, "eval_time"] = str(round(eval_time,2)) + unit
                df.loc[df["dataset"] == fname, list(metrics.keys())] = list(metrics.values())
            else:
                new_row = pd.DataFrame([{**{"dataset": fname, "size_in_MB":round(fs,2), "eval_time":str(round(eval_time,2)) + unit}, **metrics}])
                df = pd.concat([df, new_row], ignore_index=True)

            df.to_csv(csv_path, index=False)            