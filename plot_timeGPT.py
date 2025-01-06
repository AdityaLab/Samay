from nixtla import NixtlaClient
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    API_KEY = "nixak-Dcl3rmoqOEqgaNK1jd30zNLN5vhoc34loGaljdTgARJBzHeJNZuSKDwWd7azFsUGvTBoB6qjgNIp5J4k"
    client = NixtlaClient(
            api_key=API_KEY,
        )
    client.validate_api_key()
    df = pd.read_csv("data/Flu_USA/Flu_USA.csv")
    context = 40
    chunks = [df.iloc[i:i+context] for i in range(0, len(df), context)]
    chunk_ten = chunks[9]
    horizon = 4
    start_idx = chunk_ten.index[-1] + 1
    end_idx = start_idx + horizon
    df_horizon = df.iloc[start_idx:end_idx]
    forecast = client.forecast(chunk_ten, h=horizon, target_col="% WEIGHTED ILI", time_col="date", model='timegpt-1')
    pred = forecast["TimeGPT"].values
    true = df_horizon["% WEIGHTED ILI"].values
    history = chunk_ten["% WEIGHTED ILI"].values
    # save history, pred, true to a txt file
    np.savetxt("data/plot_TimeGPT.txt", history)
    np.savetxt("data/plot_TimeGPT.txt", pred)
    np.savetxt("data/plot_TimeGPT.txt", true)
    plt.plot(range(len(history)), history, label="History")
    plt.plot(range(len(history), len(history)+horizon), true, label="True")
    plt.plot(range(len(history), len(history)+horizon), pred, label="Pred")
    plt.legend()
    plt.savefig("data/plot_TimeGPT.png")
