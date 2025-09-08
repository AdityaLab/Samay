import json
import os
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from datasets import load_from_disk
from matplotlib import pyplot as plt
from tqdm import tqdm


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


def visualize(
    task_name="forecasting",
    trues=None,
    preds=None,
    history=None,
    masks=None,
    context_len=512,
    **kwargs,
):
    """
    Visualize the data.
    If task_name is "forecasting", trues, preds and history should be provided, which channel_idx and time_idx are optional.
    If task_name is "anomaly_detection", trues, preds and labels should be provided, which start and end are optional.
    If task_name is "imputation", trues, preds and masks should be provided, which channel_idx and time_idx are optional.

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
        plt.figure(figsize=(0.02 * len(history), 4))

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
        trues, preds, labels = trues[pad_len:], preds[pad_len:], labels[pad_len:]
        print(preds)
        anomaly_scores = (trues - preds) ** 2
        start = 0 if "start" not in kwargs else kwargs["start"]
        end = len(trues) if "end" not in kwargs else kwargs["end"]
        # plt.plot(trues[start:end], label="Observed", c="darkblue")
        # plt.plot(preds[start:end], label="Predicted", c="red")
        # plt.plot(anomaly_scores[start:end], label="Anomaly Score", c="black")
        # plt.legend(fontsize=16)
        # plt.show()
        fig, ax1 = plt.subplots()
        ax1.plot(trues[start:end], label="Observed", c="darkblue")
        ax1.plot(preds[start:end], label="Predicted", c="red")
        ax1.legend(fontsize=16, loc="upper left")
        ax1.set_ylabel("Value", fontsize=14)

        ax2 = ax1.twinx()
        ax2.plot(anomaly_scores[start:end], label="Anomaly Score", c="black")
        ax2.set_ylabel("Anomaly Score", fontsize=14)
        ax2.legend(fontsize=16, loc="upper right")

        plt.title(f"Anomaly Detection ({start}:{end})", fontsize=18)
        plt.xlabel("Time", fontsize=14)
        plt.show()

        best_f1, best_threshold = adjbestf1(labels, anomaly_scores)
        print(f"Best F1 Score: {best_f1:.4f} at threshold {best_threshold:.4f}")


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


def get_gifteval_datasets(path: str):
    # Get the list of hierarchical and direct datasets in the given path
    data = [x for x in os.listdir(path) if x.startswith(".") == False]
    hier, dire = [], []
    for x in data:
        if os.path.isdir(os.path.join(path, x)):
            if os.path.exists(os.path.join(path, x, "data.csv")):
                dire.append(x)
            else:
                hier.append(
                    (
                        x,
                        [
                            p
                            for p in os.listdir(os.path.join(path, x))
                            if os.path.isdir(os.path.join(path, x, p))
                            and p.startswith(".") == False
                        ],
                    )
                )

    # Get file sizes for each dataset
    fil1 = []
    for d in dire:
        d_path = os.path.join(path, d, "data.csv")
        size = os.path.getsize(d_path)
        df = pd.read_csv(d_path)
        freq = pd.infer_freq(df["timestamp"])
        fil1.append((d_path, freq, size / 1e6))

    fil2 = []
    for data, freq in hier:
        for f in freq:
            d_path = os.path.join(path, data, f, "data.csv")
            size = os.path.getsize(d_path)
            fil2.append((d_path, f, size / 1e6))
    fil = fil1 + fil2
    fil.sort(key=lambda x: x[2])
    # Create a dictionary to hold the dataset names and their frequencies
    dataset_dict = defaultdict()
    for p, freq, size in fil:
        dataset_dict[p] = (freq, size)
    # Convert the defaultdict to a regular dict
    dataset_dict = dict(dataset_dict)

    return dataset_dict, fil


def get_monash_datasets(path):
    datasets = os.listdir(path)

    # Get the filesizes
    data = []
    for x in datasets:
        d_path = os.path.join(path, x, "test", "data.csv")
        fsize = os.path.getsize(d_path) / 1e6
        data.append((x, fsize))

    data = sorted(data, key=lambda x: x[1])

    # Infer frequencies
    filesizes = []
    for i in tqdm(range(len(data)), desc="Freq inferring Monash"):
        d_path = os.path.join(path, data[i][0], "test", "data.csv")
        df = pd.read_csv(d_path)
        freq = pd.infer_freq(df["timestamp"])
        filesizes.append((data[i][0], freq, data[i][1]))

    filesizes = sorted(filesizes, key=lambda x: x[2])

    # Get dictionary for each dataset
    NAMES = defaultdict(list)
    for x in filesizes:
        NAMES[x[0]].append(x[1])

    NAMES = dict(NAMES)

    return NAMES, filesizes



def adjust_predicts(score, label, threshold=None, pred=None, calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict
    
def adjbestf1(y_true: np.array, y_scores: np.array, n_splits: int = 100):
    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_splits)
    adjusted_f1 = np.zeros(thresholds.shape)

    for i, threshold in enumerate(thresholds):
        y_pred = y_scores >= threshold
        y_pred = adjust_predicts(
            score=y_scores,
            label=(y_true > 0),
            pred=y_pred,
            threshold=None,
            calc_latency=False,
        )
        adjusted_f1[i] = f1_score(y_pred, y_true)

    best_adjusted_f1 = np.max(adjusted_f1)
    return best_adjusted_f1, thresholds[np.argmax(adjusted_f1)]

def f1_score(predict, actual):
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1

  
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
