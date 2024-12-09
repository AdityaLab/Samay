import subprocess

import numpy as np
import pandas as pd
import json
from .models.moment.momentfm.utils.data import load_from_tsfile

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


if __name__ == "__main__":
    ts_path = "/nethome/sli999/TSFMProject/src/tsfmproject/models/moment/data/ECG5000_TRAIN.ts"
    csv_path = "/nethome/sli999/TSFMProject/src/tsfmproject/models/moment/data/ECG5000_TRAIN.csv"
    # ts_to_csv(ts_path, csv_path)
    # print("Conversion complete.")
    data, labels = get_multivariate_data(pd.read_csv(csv_path))
    ts_data, ts_labels = load_from_tsfile(ts_path)
    ts_labels = np.array(ts_labels, dtype=int)
    print(data - ts_data)
    print(labels - ts_labels)
 

