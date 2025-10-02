import datetime
import gc
import os
import sys
import time

import pandas as pd
import torch

src_path = os.path.abspath(os.path.join("src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from samay.model import TimesfmModel, MomentModel, ChronosModel, ChronosBoltModel, TinyTimeMixerModel, MoiraiTSModel, LPTMModel, TimeMoEModel
from samay.dataset import TimesfmDataset, MomentDataset, ChronosDataset, ChronosBoltDataset, TinyTimeMixerDataset, MoiraiDataset, LPTMDataset, TimeMoEDataset
from samay.utils import load_args, get_gifteval_datasets, get_monash_datasets
from samay.metric import *
from samay.model import (
    ChronosBoltModel,
    ChronosModel,
    LPTMModel,
    MoiraiTSModel,
    MomentModel,
    TimesfmModel,
    TinyTimeMixerModel,
)
from samay.utils import get_gifteval_datasets, get_monash_datasets, load_args

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


SERIES = "monash" # "monash" or "gifteval"

print("Loading datasets...")
# start = time.time()
# df = pd.read_csv("data/gifteval/gifteval_datasets.csv")
# filesizes = [(x[0], x[1], x[2]) for x in df.values]
# df1 = df.groupby("datasets").agg({"freq":list}).reset_index()
# NAMES = dict(zip(df1["datasets"], df1["freq"]))
# end = time.time()
# print(f"Time taken to load datasets: {end-start:.2f} seconds")

NAMES = {
    "LOOP_SEATTLE": ["H", "5T"],
    "bitbrains_fast_storage": ["H", "5T"],
    "bitbrains_rnd": ["H", "5T"],
    "electricity": ["H", "15T"],
    "m4_daily": ["D"],
    "m4_monthly": ["M"],
    "m4_quarterly": ["Q-DEC"],
    "m4_yearly": ["A-DEC"],
    "solar": ["W", "D", "H", "10T"],
    "temperature_rain_with_missing": ["D"],
}

filesizes = [
    #   ('bitbrains_fast_storage', 'H', 15.636815),
    #   ('LOOP_SEATTLE', 'H', 27.053807),
    #   ('solar', '10T', 33.396137),
    #   ('m4_yearly', 'A-DEC', 51.396002),
    #   ('bitbrains_rnd', '5T', 63.693623),
    #   ('electricity', 'H', 110.57665),
    #   ('temperature_rain_with_missing', 'D', 113.989065),
    ("bitbrains_fast_storage", "5T", 160.063361),
    ("m4_quarterly", "Q-DEC", 163.930224),
    ("m4_daily", "D", 316.27674),
    ("LOOP_SEATTLE", "5T", 324.080655),
    ("electricity", "15T", 442.387613),
    ("m4_monthly", "M", 1025.335628),
]

MODEL_NAMES = ["moirai", "chronos", "chronosbolt", "timesfm", "moment", "ttm"]
# SERIES = "monash"


MODEL_NAMES = ["moirai", "chronos", "chronosbolt", "timesfm", "moment", "ttm", "lptm"]
MONASH_NAMES = {
    # "weather": "1D",
    "tourism_yearly": "1YE",
    "tourism_quarterly": "1Q",
    "tourism_monthly": "1M",
    "cif_2016": "1M",
    # "london_smart_meters": ["30min"],
    "australian_electricity_demand": "30min",
    # "wind_farms_minutely": ["1min"],
    "bitcoin": "1D",
    "pedestrian_counts": "1h",
    "vehicle_trips": "1D",
    "kdd_cup_2018": "1H",
    "nn5_daily": "1D",
    "nn5_weekly": "1W",
    # "kaggle_web_traffic": ["1D"],
    # "kaggle_web_traffic_weekly": ["1W"],
    "solar_10_minutes": "10min",
    "solar_weekly": "1W",
    "car_parts": "1M",
    "fred_md": "1M",
    "traffic_hourly": "1h",
    "traffic_weekly": "1W",
    "hospital": "1M",
    "covid_deaths": "1D",
    "sunspot": "1D",
    "saugeenday": "1D",
    "us_births": "1D",
    "solar_4_seconds": "4s",
    "wind_4_seconds": "4s",
    "rideshare": "1h",
    "oikolab_weather": "1h",
    "temperature_rain": "1D",
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
    "temperature_rain": 30,
}

MODEL_CONTEXT_LEN = {"timesfm": 32, "moment": 512, "chronos": 512}

start = time.time()
if SERIES == "gifteval":
    # Load the datasets from the Gifteval dataset
    NAMES = get_gifteval_datasets("data/gifteval")
elif SERIES == "monash":
    # Load the datasets from the Monash dataset
    NAMES = get_monash_datasets("data/monash")

end = time.time()
print(NAMES)

print(f"Time taken to load datasets: {end - start:.2f} seconds")


def calc_pred_and_context_len(freq):
    # split feq into base and multiplier
    base = freq[-1]
    mult = int(freq[:-1]) if len(freq) > 1 else 1
    if base == "Y":
        pred_len = 4
        context_len = MODEL_CONTEXT_LEN["timesfm"]
    elif base == "Q":
        pred_len = 4
        context_len = MODEL_CONTEXT_LEN["timesfm"]
    elif base == "M":
        pred_len = 12 // mult
        context_len = MODEL_CONTEXT_LEN["timesfm"]
    elif base == "W":
        pred_len = 4 // mult
        context_len = MODEL_CONTEXT_LEN["timesfm"]
    elif base == "D":
        pred_len = 7 // mult
        context_len = MODEL_CONTEXT_LEN["timesfm"]
    elif base == "H":
        pred_len = 24 // mult
        context_len = 2 * MODEL_CONTEXT_LEN["timesfm"]
    elif base == "S":
        pred_len = 60 // mult
        context_len = 4 * MODEL_CONTEXT_LEN["timesfm"]
    else:
        raise ValueError(f"Invalid frequency: {freq}")
    return pred_len, context_len


if __name__ == "__main__":
    
    for model_name in ["moirai2"]:
        print(f"Evaluating model: {model_name}")
        # create csv file for leaderboard if not already created
        csv_path = f"leaderboard/{model_name}.csv"
        if SERIES == "monash":
            csv_path = f"leaderboard/monash_{model_name}.csv"
        if not os.path.exists(csv_path):
            print(f"Creating leaderboard csv file: {csv_path}")
            df = pd.DataFrame(
                columns=[
                    "dataset",
                    "size_in_MB",
                    "eval_time",
                    "mse",
                    "mae",
                    "mase",
                    "mape",
                    "rmse",
                    "nrmse",
                    "smape",
                    "msis",
                    "nd",
                    "mwsq",
                    "crps",
                ]
            )
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
        elif model_name == "moirai2":
            arg_path = "config/moirai2.json"
            args = load_args(arg_path)
        elif model_name == "lptm":
            arg_path = "config/lptm.json"
            args = load_args(arg_path)
        elif model_name == "timemoe":
            arg_path = "config/timemoe.json"
            args = load_args(arg_path)

        mod_start = time.time()
        mod_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for fpath, (freq, fs) in NAMES.items():
            fname = fpath.split("/")[2]
            print(f"Model eval started at: {mod_timestamp}")
            print(
                f"Evaluating {fname} ({freq}) started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            # Adjust the context and prediction length based on the frequency

            # pred_len, context_len = calc_pred_and_context_len(freq)
            pred_len, context_len = 96, 512
            if SERIES == "monash":
                pred_len = MONASH_SETTINGS[fname]

            if model_name == "timesfm":
                args["config"]["horizon_len"] = pred_len
                args["config"]["context_len"] = context_len
            elif model_name == "moment":
                args["config"]["forecast_horizon"] = pred_len
            elif model_name == "ttm":
                args["config"]["horizon_len"] = pred_len
                args["config"]["context_len"] = context_len
            elif model_name == "moirai" or model_name == "moirai2":
                args["config"]["horizon_len"] = pred_len
                args["config"]["context_len"] = context_len

            dataset_path = fpath

            if model_name == "timesfm":
                dataset = TimesfmDataset(
                    datetime_col="timestamp",
                    path=dataset_path,
                    mode="test",
                    context_len=args["config"]["context_len"],
                    horizon_len=args["config"]["horizon_len"],
                    boundaries=(-1, -1, -1),
                    batchsize=64,
                )
                args["config"]["horizon_len"] = dataset.horizon_len
                model = TimesfmModel(**args)
                start = time.time()
                metrics = model.evaluate(dataset)
                print("Metrics: ", metrics)
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(
                    f"Time taken for evaluation of {fname}: {end - start:.2f} seconds"
                )

                del model
                torch.cuda.empty_cache()
                gc.collect()

            elif model_name == "moment":
                args["config"]["task_name"] = "forecasting"
                train_dataset = MomentDataset(
                    datetime_col="timestamp",
                    path=dataset_path,
                    mode="train",
                    horizon_len=args["config"]["forecast_horizon"],
                    normalize=False,
                )
                dataset = MomentDataset(
                    datetime_col="timestamp",
                    path=dataset_path,
                    mode="test",
                    horizon_len=args["config"]["forecast_horizon"],
                    normalize=False,
                    boundaries=[-1, -1, -1],
                )
                args["config"]["forecast_horizon"] = dataset.forecast_horizon
                model = MomentModel(**args)
                finetuned_model = model.finetune(train_dataset, task_name="forecasting")
                start = time.time()
                metrics = model.evaluate(dataset, task_name="forecasting")
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(
                    f"Time taken for evaluation of {fname}: {end - start:.2f} seconds"
                )
                print(metrics)

                del model
                del finetuned_model
                del dataset
                del train_dataset
                torch.cuda.empty_cache()
                gc.collect()

            elif model_name == "chronos":
                dataset_config = load_args("config/chronos_dataset.json")
                dataset_config["context_length"] = context_len
                dataset_config["prediction_length"] = pred_len
                dataset = ChronosDataset(
                    datetime_col="timestamp",
                    path=dataset_path,
                    mode="test",
                    config=dataset_config,
                    batch_size=4,
                    boundaries=[-1, -1, -1],
                )
                args["config"]["context_length"] = dataset.horizon_len
                model = ChronosModel(**args)
                start = time.time()
                metrics = model.evaluate(
                    dataset,
                    horizon_len=dataset.horizon_len,
                    quantile_levels=[0.1, 0.5, 0.9],
                )
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(
                    f"Time taken for evaluation of {fname}: {end - start:.2f} seconds"
                )

                del model
                del dataset
                torch.cuda.empty_cache()
                gc.collect()

            elif model_name == "chronosbolt":
                repo = "amazon/chronos-bolt-small"
                model = ChronosBoltModel(repo=repo)
                dataset = ChronosBoltDataset(
                    datetime_col="timestamp",
                    path=dataset_path,
                    mode="test",
                    batch_size=8,
                    context_len=context_len,
                    horizon_len=pred_len,
                    boundaries=[-1, -1, -1],
                )
                start = time.time()
                metrics = model.evaluate(
                    dataset,
                    horizon_len=dataset.horizon_len,
                    quantile_levels=[0.1, 0.5, 0.9],
                )
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(
                    f"Time taken for evaluation of {fname}: {end - start:.2f} seconds"
                )

                del model
                del dataset
                torch.cuda.empty_cache()
                gc.collect()

            elif model_name == "ttm":
                dataset = TinyTimeMixerDataset(
                    datetime_col="timestamp",
                    path=dataset_path,
                    mode="test",
                    context_len=context_len,
                    horizon_len=pred_len,
                    boundaries=[-1, -1, -1],
                )
                args["config"]["horizon_len"] = dataset.horizon_len
                model = TinyTimeMixerModel(**args)
                start = time.time()
                metrics = model.evaluate(dataset)
                end = time.time()
                print("Metrics: ", metrics)
                print(f"Size of dataset: {fs:.2f} MB")
                print(
                    f"Time taken for evaluation of {fname}: {end - start:.2f} seconds"
                )

                del model
                del dataset
                torch.cuda.empty_cache()
                gc.collect()

            elif model_name == "moirai":
                model = MoiraiTSModel(**args)
                dataset = MoiraiDataset(
                    name=fname,
                    datetime_col="timestamp",
                    freq=freq,
                    path=dataset_path,
                    mode="test",
                    context_len=context_len,
                    horizon_len=pred_len,
                )

                start = time.time()
                metrics = model.evaluate(dataset, leaderboard=True)
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(
                    f"Time taken for evaluation of {fname}: {end - start:.2f} seconds"
                )

                del model
                del dataset
                torch.cuda.empty_cache()
                gc.collect()
            
            elif model_name == "moirai2":
                model = MoiraiTSModel(**args)
                dataset = MoiraiDataset(
                    name=fname,
                    datetime_col="timestamp",
                    freq=freq,
                    path=dataset_path,
                    mode="test",
                    context_len=context_len,
                    horizon_len=pred_len,
                )

                start = time.time()
                metrics = model.evaluate(dataset, leaderboard=True)
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(
                    f"Time taken for evaluation of {fname}: {end - start:.2f} seconds"
                )

                del model
                del dataset
                torch.cuda.empty_cache()
                gc.collect()

            elif model_name == "lptm":
                args["config"]["task_name"] = "forecasting2"
                dataset = LPTMDataset(
                    name=fname,
                    datetime_col="timestamp",
                    task_name="forecasting2",
                    path=dataset_path,
                    mode="test",
                    seq_len=context_len,
                    horizon=pred_len,
                )
                args["config"]["forecast_horizon"] = dataset.forecast_horizon
                model = LPTMModel(**args)
                start = time.time()
                metrics = model.evaluate(dataset, task_name="forecasting2")
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(
                    f"Time taken for evaluation of {fname}: {end - start:.2f} seconds"
                )

                del model
                del dataset
                torch.cuda.empty_cache()
                gc.collect()


            elif model_name == "timemoe":
                dataset = TimeMoEDataset(name=fname, datetime_col='timestamp', freq=freq, batch_size=64,
                                        path=dataset_path, mode='test', context_len=context_len, horizon_len=pred_len, boundaries=[-1, -1, -1])
                args["config"]["horizon_len"] = dataset.horizon_len
                model = TimeMoEModel(**args)
                start = time.time()
                metrics = model.evaluate(dataset)
                end = time.time()
                print(f"Size of dataset: {fs:.2f} MB")
                print(f"Time taken for evaluation of {fname}: {end-start:.2f} seconds")

                del model
                del dataset
                torch.cuda.empty_cache()
                gc.collect()

            print("Evaluation done!")

            eval_time = end - start
            unit = "s"
            if eval_time > 1000:  # convert to minutes
                eval_time = eval_time / 60
                unit = "m"

            df = pd.read_csv(csv_path)
            row_name = fname + " (" + freq + ")"
            if row_name in df["dataset"].values:
                df.loc[df["dataset"] == row_name, "size_in_MB"] = round(fs, 2)
                df.loc[df["dataset"] == row_name, "eval_time"] = (
                    str(round(eval_time, 2)) + unit
                )
                df.loc[df["dataset"] == row_name, list(metrics.keys())] = list(
                    metrics.values()
                )
            else:
                new_row = pd.DataFrame(
                    [
                        {
                            **{
                                "dataset": row_name,
                                "size_in_MB": round(fs, 2),
                                "eval_time": str(round(eval_time, 2)) + unit,
                            },
                            **metrics,
                        }
                    ]
                )
                df = pd.concat([df, new_row], ignore_index=True)

            df.to_csv(csv_path, index=False)
        mod_end = time.time()
        print(f"Time taken for model {model_name}: {mod_end - mod_start:.2f} seconds")
        mod_timestamp[model_name] = round(mod_end - mod_start, 2)

    print("All models evaluated!")
    print("Model evaluation times: ", mod_timestamp)
