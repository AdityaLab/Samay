import os 
import sys
import numpy as np
import pandas as pd
import time
import datetime

src_path = os.path.abspath(os.path.join("src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from samay.model import TimesfmModel, MomentModel, ChronosModel, ChronosBoltModel, TinyTimeMixerModel, MoiraiTSModel
from samay.dataset import TimesfmDataset, MomentDataset, ChronosDataset, ChronosBoltDataset, TinyTimeMixerDataset, MoiraiDataset
from samay.utils import load_args, get_gifteval_datasets
from samay.metric import *


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

print("Loading datasets...")
# start = time.time()
# df = pd.read_csv("data/monash/monash_datasets.csv")
# filesizes = [(x[0], x[1], x[2]) for x in df.values]
# df1 = df.groupby("datasets").agg({"freq":list}).reset_index()
# NAMES = dict(zip(df1["datasets"], df1["freq"]))
# end = time.time()
# print(f"Time taken to load datasets: {end-start:.2f} seconds")

NAMES = {'kaggle_web_traffic': ['D'],
 'london_smart_meters': ['30T'],
 'solar_4_seconds': ['4S'],
 'weather': ['D'],
 'wind_4_seconds': ['4S'],
 'wind_farms_minutely': ['T']}

filesizes = [('solar_4_seconds', '4S', 181.726275),
 ('wind_4_seconds', '4S', 184.145404),
 ('kaggle_web_traffic', 'D', 639.06982),
 ('wind_farms_minutely', 'T', 835.989173),
 ('weather', 'D', 1180.743458),
 ('london_smart_meters', '30T', 3320.44015)]

MODEL_NAMES = ["moirai", "chronos", "chronosbolt", "timesfm", "moment", "ttm"]
MONASH_NAMES = {
    # "weather": "1D",
    "tourism_yearly": ["1YE"],
    "tourism_quarterly": ["1Q"],
    "tourism_monthly": ["1M"],
    "cif_2016": ["1M"],
    # "london_smart_meters": ["30min"],
    "australian_electricity_demand": ["30min"],
    # "wind_farms_minutely": ["1min"],
    "bitcoin": ["1D"],
    "pedestrian_counts": ["1h"],
    "vehicle_trips": ["1D"],
    "kdd_cup_2018": ["1H"],
    "nn5_daily": ["1D"],
    "nn5_weekly": ["1W"],
    # "kaggle_web_traffic": ["1D"],
    # "kaggle_web_traffic_weekly": ["1W"],
    "solar_10_minutes": ["10min"],
    "solar_weekly": ["1W"],
    "car_parts": ["1M"],
    "fred_md": ["1M"],
    "traffic_hourly": ["1h"],
    "traffic_weekly": ["1W"],
    "hospital": ["1M"],
    "covid_deaths": ["1D"],
    "sunspot": ["1D"],
    "saugeenday": ["1D"],
    "us_births": ["1D"],
    "solar_4_seconds": ["4s"],
    "wind_4_seconds": ["4s"],
    "rideshare": ["1h"],
    "oikolab_weather": ["1h"],
    "temperature_rain": ["1D"]
}

MONASH_SETTINGS = {
    "weather": 30,
    "tourism_yearly": 4,
    "tourism_quarterly": 8,
    "tourism_monthly": 24,
    "cif_2016": 12,
    "london_smart_meters": 60,
    "australian_electricity_demand": 60,
    "wind_farms_minutely": 60,
    "bitcoin": 30,
    "pedestrian_counts": 48,
    "vehicle_trips": 30,
    "kdd_cup_2018": 48,
    "nn5_daily": 56,
    "nn5_weekly": 8,
    "kaggle_web_traffic": 59,
    "kaggle_web_traffic_weekly": 8,
    "solar_10_minutes": 60,
    "solar_weekly": 5,
    "car_parts": 12,
    "fred_md": 12,
    "traffic_hourly": 48,
    "traffic_weekly": 8,
    "hospital": 12,
    "covid_deaths": 30,
    "sunspot": 30,
    "saugeenday": 30,
    "us_births": 30,
    "solar_4_seconds": 60,
    "wind_4_seconds": 60,
    "rideshare": 48,
    "oikolab_weather": 48,
    "temperature_rain": 30
}

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
    mod_times = {}
    for model_name in ["moirai"]:
        print(f"Evaluating model: {model_name}")
        # create csv file for leaderboard if not already created
        csv_path = f"leaderboard/monash_{model_name}.csv"
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

        mod_start = time.time()
        mod_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for fname, freq, fs in filesizes:
            print(f"Model eval started at: {mod_timestamp}")
            print(f"Evaluating {fname} ({freq}) started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            # Adjust the context and prediction length based on the frequency

            # pred_len, context_len = calc_pred_and_context_len(freq)
            pred_len, context_len = MONASH_SETTINGS[fname], 512
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
                dataset_path = f"data/monash/{fname}/test/data.csv"
            else:
                dataset_path = f"data/monash/{fname}/{freq}/test/data.csv"
            
            if model_name == "timesfm":
                
                dataset = TimesfmDataset(datetime_col='timestamp', path=dataset_path, mode='test', context_len=args["config"]["context_len"], horizon_len=args["config"]["horizon_len"], boundaries=(-1, -1, -1), batchsize=64)
                args["config"]["horizon_len"] = dataset.horizon_len
                model = TimesfmModel(**args)
                start = time.time()
                metrics = model.evaluate(dataset)
                print("Metrics: ", metrics)
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(f"Time taken for evaluation of {fname}: {end-start:.2f} seconds")

            elif model_name == "moment":
                
                args["config"]["task_name"] = "forecasting"
                train_dataset = MomentDataset(datetime_col='timestamp', path=dataset_path, mode='train', horizon_len=args["config"]["forecast_horizon"], normalize=False)
                dataset = MomentDataset(datetime_col='timestamp', path=dataset_path, mode='test', horizon_len=args["config"]["forecast_horizon"], normalize=False, boundaries=[-1, -1, -1])
                args["config"]["forecast_horizon"] = dataset.forecast_horizon
                model = MomentModel(**args)
                finetuned_model = model.finetune(train_dataset, task_name="forecasting")
                start = time.time()
                metrics = model.evaluate(dataset, task_name="forecasting")
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(f"Time taken for evaluation of {fname}: {end-start:.2f} seconds")
                print(metrics)

            elif model_name == "chronos":
                
                dataset_config = load_args("config/chronos_dataset.json")
                dataset_config["context_length"] = context_len
                dataset_config["prediction_length"] = pred_len
                dataset = ChronosDataset(datetime_col='timestamp', path=dataset_path, mode='test', config=dataset_config, batch_size=4, boundaries=[-1, -1, -1])
                args["config"]["context_length"] = dataset.horizon_len
                model = ChronosModel(**args)
                start = time.time()
                metrics = model.evaluate(dataset, horizon_len=dataset.horizon_len, quantile_levels=[0.1, 0.5, 0.9])
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(f"Time taken for evaluation of {fname}: {end-start:.2f} seconds")

            elif model_name == "chronosbolt":
                repo = "amazon/chronos-bolt-small"
                model = ChronosBoltModel(repo=repo)
                dataset = ChronosBoltDataset(datetime_col='timestamp', path=dataset_path, mode='test', batch_size=8, context_len=context_len, horizon_len=pred_len, boundaries=[-1, -1, -1])
                start = time.time()
                metrics = model.evaluate(dataset, horizon_len=dataset.horizon_len, quantile_levels=[0.1, 0.5, 0.9])
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(f"Time taken for evaluation of {fname}: {end-start:.2f} seconds")

            elif model_name == "ttm":
                
                dataset = TinyTimeMixerDataset(datetime_col='timestamp', path=dataset_path, mode='test', context_len=context_len, horizon_len=pred_len, boundaries=[-1, -1, -1])
                args["config"]["horizon_len"] = dataset.horizon_len
                model = TinyTimeMixerModel(**args)
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
            row_name = fname + ' (' + freq + ')'
            if row_name in df["dataset"].values:
                df.loc[df["dataset"] == row_name, "size_in_MB"] = round(fs,2)
                df.loc[df["dataset"] == row_name, "eval_time"] = str(round(eval_time,2)) + unit
                df.loc[df["dataset"] == row_name, list(metrics.keys())] = list(metrics.values())
            else:
                new_row = pd.DataFrame([{**{"dataset": row_name, "size_in_MB":round(fs,2), "eval_time":str(round(eval_time,2)) + unit}, **metrics}])
                df = pd.concat([df, new_row], ignore_index=True)

            df.to_csv(csv_path, index=False)
        mod_end = time.time()
        print(f"Time taken for model {model_name}: {mod_end-mod_start:.2f} seconds")
        mod_times[model_name] = round(mod_end - mod_start,2)
    
    print("All models evaluated!")
    print("Model evaluation times: ", mod_times)