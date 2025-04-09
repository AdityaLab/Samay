import json
import subprocess
import os

import numpy as np
import pandas as pd
import yaml

from datasets import load_from_disk 
from matplotlib import pyplot as plt
from collections import defaultdict

def get_least_used_gpu():
    """Get the least used GPU device."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE,
        )
        gpu_memory_used = [
            int(x) for x in result.stdout.decode("utf-8").strip().split("\n")
        ]
        return np.argmin(gpu_memory_used)
    except Exception:
        return -1


def ts_to_csv(ts_file, csv_file, replace_missing_vals_with="NaN"):
    """
    Convert a .ts file to a .csv file.

    Parameters
    ----------
    ts_file : str
        Path to the .ts file.
    csv_file : str
        Path to the output .csv file.
    replace_missing_vals_with : str, default="NaN"
        Replace missing values in the .ts file with this value.
    """
    data = []
    labels = []

    with open(ts_file, "r", encoding="utf-8") as file:
        in_data_section = False
        for line in file:
            line = line.strip()
            # Skip metadata and comments
            if not in_data_section:
                if line.lower() == "@data":
                    in_data_section = True
                continue

            # Process data section
            line = line.replace("?", replace_missing_vals_with)
            channels = line.split(":")
            if len(channels) > 1:  # Multivariate or labeled
                *features, label = channels
                labels.append(label)
            else:  # Univariate or unlabeled
                features = channels
                labels.append(None)

            # Convert features to a flat list of floats
            flattened_features = []
            column_name = []
            for j, channel in enumerate(features):
                flattened_features.extend([float(x) for x in channel.split(",")])
                # add column names for each channel
                column_name.extend(
                    [f"channel_{j}_time_{i}" for i in range(len(channel.split(",")))]
                )

            data.append(flattened_features)

    # Create DataFrame
    df = pd.DataFrame(data)
    df.columns = column_name
    if any(labels):  # If labels exist, add as the last column
        df["label"] = labels

    # Save to CSV
    df.to_csv(csv_file, index=False)


def get_multivariate_data(dataframe, label_col="label"):
    """
    Get multivariate data from a .csv file.
    Input:
    dataframe: pd.DataFrame
    Output:
    data: np.ndarray
    labels: np.ndarray
    """
    labels = dataframe[label_col].values
    dataframe.drop(columns=[label_col], inplace=True)
    num_channels = int(dataframe.columns[-1].split("_")[1]) + 1
    num_timesteps = int(dataframe.columns[-1].split("_")[-1]) + 1
    # reshape data into (num_samples, num_channels, num_timesteps)
    data = dataframe.values.reshape(-1, num_channels, num_timesteps)
    return data, labels


def load_args(file_path):
    """
    Load arguments from a file.
    """
    with open(file_path, "r") as file:
        return json.load(file)


def arrow_to_csv(arrow_dir, freq=None):
    data = load_from_disk(arrow_dir)
    df = data.to_pandas()
    start_date = df["start"].iloc[0]
    df = df.drop(columns=["start"])
    print(start_date)
    df_expanded = df.explode("target", ignore_index=True)
    df_expanded = df_expanded[["target", "item_id"]]
    # if the target column is not numeric but an array, explode again
    if isinstance(df_expanded["target"].iloc[0], np.ndarray):
        print(df_expanded["target"].iloc[0].shape)
        df_expanded["group_id"] = ["T" + str(i) for i in range(1, len(df_expanded) + 1)]
        df_expanded = df_expanded.explode("target", ignore_index=True)
        df_expanded["item_id"] = df_expanded["group_id"]
        df_expanded.drop(columns=["group_id"], inplace=True)
    df_expanded.infer_objects()
    print(df_expanded.head())
    max_length = max(
        group["target"].size for _, group in df_expanded.groupby("item_id")
    )
    pivot_df = pd.DataFrame(
        {
            item: group["target"].tolist() + [0] * (max_length - len(group["target"]))
            for item, group in df_expanded.groupby("item_id")
        }
    )
    pivot_df["timestamp"] = pd.date_range(
        start=start_date, periods=len(pivot_df), freq=freq
    )
    csv_file = arrow_dir + "/data.csv"
    pivot_df.to_csv(csv_file, index=False)
    print(f"Conversion complete for {arrow_dir}.")


def visualize(task_name="forecasting", trues=None, preds=None, history=None, masks=None, context_len=512, **kwargs):
    """
    Visualize the data.
    If task_name is "forecasting", trues, preds, and history should be provided, which channel_idx and time_idx are optional.
    If task_name is "anomaly_detection", trues and preds should be provided, which start and end are optional.
    If task_name is "imputation", trues, preds, and masks should be provided, which channel_idx and time_idx are optional.

    channel_idx correpsonds to a specific variate (column) in the dataset.
    time_idx corresponds to a specific window (context + horizon) in the augmented dataset.
    """
    trues = np.array(trues)
    preds = np.array(preds)
    if task_name == "forecasting":
        channel_idx = (
            np.random.randint(0, trues.shape[1])
            if "channel_idx" not in kwargs
            else kwargs["channel_idx"]
        )  # variate index
        time_idx = (
            np.random.randint(0, trues.shape[0])
            if "time_idx" not in kwargs
            else kwargs["time_idx"]
        )  # a specific series of context + prediction
        dataset = kwargs["dataset"] if "dataset" in kwargs else None
        freq = kwargs["freq"] if "freq" in kwargs else None

        if isinstance(history, np.ndarray):
            print(history.shape)
            history = history[time_idx, channel_idx, -context_len:]
        elif isinstance(history, list):
            history = history[time_idx][channel_idx][-context_len:]
        true = trues[time_idx, channel_idx, :]
        pred = preds[time_idx, channel_idx, :]

        # Set figure size proportional to the number of forecasts
        plt.figure(figsize=(0.2 * len(history), 4))

        # Plotting the first time series from history
        plt.plot(
            range(len(history)),
            history,
            label=f"History ({len(history)} timesteps)",
            c="darkblue",
        )

        offset = len(history)
        plt.plot(
            range(offset, offset + len(true)),
            true,
            label=f"Ground Truth ({len(true)} timesteps)",
            color="darkblue",
            linestyle="--",
            alpha=0.5,
        )
        plt.plot(
            range(offset, offset + len(pred)),
            pred,
            label=f"Forecast ({len(pred)} timesteps)",
            color="red",
            linestyle="--",
        )

        plt.title(
            f"{dataset} ({freq}) -- (window={time_idx}, variate index={channel_idx})",
            fontsize=18,
        )
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.legend(fontsize=14, loc="upper left")
        plt.show()

    elif task_name == "detection":
        anomaly_scores = (trues - preds) ** 2
        start = 0 if "start" not in kwargs else kwargs["start"]
        end = 1000 if "end" not in kwargs else kwargs["end"]
        plt.plot(trues[start:end], label="Observed", c="darkblue")
        plt.plot(preds[start:end], label="Predicted", c="red")
        plt.plot(anomaly_scores[start:end], label="Anomaly Score", c="black")
        plt.legend(fontsize=16)
        plt.show()

    elif task_name == "imputation":
        time_idx = (
            np.random.randint(0, trues.shape[0])
            if "time_idx" not in kwargs
            else kwargs["time_idx"]
        )
        channel_idx = (
            np.random.randint(0, trues.shape[1])
            if "channel_idx" not in kwargs
            else kwargs["channel_idx"]
        )
        fig, axs = plt.subplots(2, 1, figsize=(10, 5))
        axs[0].set_title(f"Channel={channel_idx}")
        axs[0].plot(
            trues[time_idx, channel_idx, :].squeeze(),
            label="Ground Truth",
            c="darkblue",
        )
        axs[0].plot(
            preds[time_idx, channel_idx, :].squeeze(), label="Predictions", c="red"
        )
        axs[0].legend(fontsize=16)

        axs[1].imshow(
            np.tile(masks[np.newaxis, time_idx, channel_idx], reps=(8, 1)),
            cmap="binary",
        )
        plt.show()


def read_yaml(file_path):
    """
    Read a YAML file.
    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def prep_finetune_config(file_path: str = None, config: dict = None):
    """
    Prepare the finetune configuration.
    """
    assert file_path is not None or config is not None, (
        "Either file_path or config must be provided."
    )

    if config is None:
        config = read_yaml(file_path)

    return {
        "batch_size": config["train_dataloader"]["batch_size"],
        "max_epochs": config["trainer"]["max_epochs"],
        "seed": config["seed"],
        "tf32": config["tf32"],
        "mod_torch": {
            k: v
            for k, v in config["trainer"].items()
            if k != "_target_" and type(v) not in [dict, list]
        },
    }


def get_gifteval_datasets(path:str):
    # Get the list of hierarchical and direct datasets in the given path
    data = [x for x in os.listdir(path) if x.startswith(".")==False]
    hier, dire = [], []
    for x in data:
        if os.path.isdir(os.path.join(path, x)):
            if os.path.exists(os.path.join(path, x, "data.csv")):
                dire.append(x)
            else:
                hier.append((x, [p for p in os.listdir(os.path.join(path, x)) if os.path.isdir(os.path.join(path, x, p)) and p.startswith(".")==False]))
    
    # Get file sizes for each dataset
    fil1 = []
    for d in dire:
        d_path = os.path.join(path, d, "data.csv")
        size = os.path.getsize(d_path)
        df = pd.read_csv(d_path)
        freq = pd.infer_freq(df["timestamp"])
        fil1.append((d_path, freq, size/1e6))
    
    fil2 = []
    for data,freq in hier:
        for f in freq:
            d_path = os.path.join(path, data, f, "data.csv")
            size = os.path.getsize(d_path)
            fil2.append((d_path, f, size/1e6))
    fil = fil1 + fil2
    fil.sort(key=lambda x: x[2])
    # Create a dictionary to hold the dataset names and their frequencies
    dataset_dict = defaultdict()
    for p, freq, size in fil:
        dataset_dict[p] = (freq, size)
    # Convert the defaultdict to a regular dict
    dataset_dict = dict(dataset_dict)

    return dataset_dict

def get_monash_datasets(path:str, config:dict, setting:dict):
    dataset_names = config.keys()
    dataset_paths = [path + "/" + name + "/test/data.csv" for name in dataset_names]
    # Get the frequencies for each dataset
    dataset_freqs = [config[name] for name in dataset_names]
    dataset_horizons = [setting[name] for name in dataset_names]
    
    # sort the datasets by size, ascending
    dataset_sizes = []
    for p in dataset_paths:
        size = os.path.getsize(p)
        dataset_sizes.append(size/1e6)
    dataset_paths, dataset_freqs, dataset_horizons, dataset_sizes = zip(*sorted(zip(dataset_paths, dataset_freqs, dataset_horizons, dataset_sizes), key=lambda x: x[3]))
    # Create a dictionary to hold the dataset names and their frequencies
    dataset_dict = defaultdict()
    fil = zip(dataset_paths, dataset_freqs, dataset_horizons, dataset_sizes)
    # turn fil into a list
    fil = list(fil)
    for p, freq, horizon, size in fil:
        dataset_dict[p] = (freq, horizon, size)
    # Convert the defaultdict to a regular dict
    dataset_dict = dict(dataset_dict)

    return dataset_dict

  
if __name__ == "__main__":
    # ts_path = "/nethome/sli999/TSFMProject/src/tsfmproject/models/moment/data/ECG5000_TRAIN.ts"
    # csv_path = "/nethome/sli999/TSFMProject/src/tsfmproject/models/moment/data/ECG5000_TRAIN.csv"
    # # ts_to_csv(ts_path, csv_path)
    # # print("Conversion complete.")
    # data, labels = get_multivariate_data(pd.read_csv(csv_path))
    # ts_data, ts_labels = load_from_tsfile(ts_path)
    # ts_labels = np.array(ts_labels, dtype=int)
    # print(data - ts_data)
    # print(labels - ts_labels)
    arrow_dir = "/nethome/sli999/TSFMProject/data/monash/wind_farms_minutely/train"
    # arrow_to_csv(arrow_dir)
    # print("Conversion complete.")
    csv_file = arrow_dir + "/data.csv"
    df = pd.read_csv(csv_file)
    print(df.head())
