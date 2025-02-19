import subprocess

import numpy as np
import pandas as pd
import json
from .models.moment.momentfm.utils.data import load_from_tsfile
from datasets import load_from_disk 
from matplotlib import pyplot as plt

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
                column_name.extend([f"channel_{j}_time_{i}" for i in range(len(channel.split(",")))])

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

def arrow_to_csv(arrow_dir):
    data = load_from_disk(arrow_dir)
    df = data.to_pandas()
    start_date = df['start'].iloc[0]
    df = df.drop(columns=['start'])
    df_expanded = df.explode('target', ignore_index=True)
    df_expanded.infer_objects()
    max_length = max(group['target'].size for _, group in df_expanded.groupby('item_id'))
    pivot_df = pd.DataFrame({item: group['target'].tolist() + [0] * (max_length - len(group['target'])) for item, group in df_expanded.groupby('item_id')})
    pivot_df['timestamp'] = pd.date_range(start=start_date, periods=len(pivot_df), freq='D')
    csv_file = arrow_dir + "/data.csv"
    pivot_df.to_csv(csv_file, index=False)
    print(f"Conversion complete for {arrow_dir}.")


def visualize(task_name="forecasting", trues=None, preds=None, history=None, masks=None, **kwargs):
    """
    Visualize the data.
    If task_name is "forecasting", trues, preds, and history should be provided, which channel_idx and time_idx are optional.
    If task_name is "anomaly_detection", trues and preds should be provided, which start and end are optional.
    If task_name is "imputation", trues, preds, and masks should be provided, which channel_idx and time_idx are optional.
    """
    trues = np.array(trues)
    preds = np.array(preds)
    if task_name == "forecasting":
        histories = np.array(history)
        channel_idx = np.random.randint(0, trues.shape[1]) if "channel_idx" not in kwargs else kwargs["channel_idx"]
        time_idx = np.random.randint(0, trues.shape[0]) if "time_idx" not in kwargs else kwargs["time_idx"] 
        history = histories[time_idx, channel_idx, :] 
        true = trues[time_idx, channel_idx, :]
        pred = preds[time_idx, channel_idx, :]

        num_history = len(history)

        # Set figure size proportional to the number of forecasts
        plt.figure(figsize=(0.02 * num_history, 5))

        # Plotting the first time series from history
        plt.plot(range(len(history)), history, label=f'History ({len(history)} timesteps)', c='darkblue')

        offset = len(history)
        plt.plot(range(offset, offset + len(true)), true, label=f'Ground Truth ({len(true)} timesteps)', color='darkblue', linestyle='--', alpha=0.5)
        plt.plot(range(offset, offset + len(pred)), pred, label=f'Forecast ({len(pred)} timesteps)', color='red', linestyle='--')

        plt.title(f"(idx={time_idx}, channel={channel_idx})", fontsize=18)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.legend(fontsize=14)
        plt.show()

    elif task_name == "detection":
        anomaly_scores = (trues - preds)**2
        start = 0 if "start" not in kwargs else kwargs["start"]
        end = 1000 if "end" not in kwargs else kwargs["end"]
        plt.plot(trues[start:end], label="Observed", c='darkblue')
        plt.plot(preds[start:end], label="Predicted", c='red')
        plt.plot(anomaly_scores[start:end], label="Anomaly Score", c='black')
        plt.legend(fontsize=16)
        plt.show()

    elif task_name == "imputation":
        time_idx = np.random.randint(0, trues.shape[0]) if "time_idx" not in kwargs else kwargs["time_idx"]
        channel_idx = np.random.randint(0, trues.shape[1]) if "channel_idx" not in kwargs else kwargs["channel_idx"]
        fig, axs = plt.subplots(2, 1, figsize=(10, 5))
        axs[0].set_title(f"Channel={channel_idx}")
        axs[0].plot(trues[time_idx, channel_idx, :].squeeze(), label='Ground Truth', c='darkblue')
        axs[0].plot(preds[time_idx, channel_idx, :].squeeze(), label='Predictions', c='red')
        axs[0].legend(fontsize=16)

        axs[1].imshow(np.tile(masks[np.newaxis, time_idx, channel_idx], reps=(8, 1)), cmap='binary')
        plt.show()

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
    
 




    