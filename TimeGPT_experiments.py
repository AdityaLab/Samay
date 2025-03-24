from nixtla import NixtlaClient
import pandas as pd
import numpy as np
from datasets import load_from_disk


if __name__ == "__main__":
    API_KEY = ["nixak-Dcl3rmoqOEqgaNK1jd30zNLN5vhoc34loGaljdTgARJBzHeJNZuSKDwWd7azFsUGvTBoB6qjgNIp5J4k",
               "nixak-lgkeiACnUJx7jbslOQwP4qgYByjbKkmHG2iCiLjo2Ymy7B8tEEXyo26JFzKToXDWwAvK3i8u98uxnDph",
               "nixak-AsJr2gE9btKpfOCp654eKQX47ALZwWdoArI5gZNN5LQMIGDeO1SFeEsSpceB2fFdMKYtmxomU47vg8N4",
               "nixak-IlppDwy73vQjqbEzxMGrMx0PGYcn358jCrzEBYxv1OGjtcvSy3YIayC5wmNsFtDnOTXCF8vnwvKsEWU1"]
    for horizon in [1, 2, 3, 4]:
        client = NixtlaClient(
            api_key=API_KEY[horizon-1],
        )
        client.validate_api_key()
        df = pd.read_csv("data/Flu_USA/Flu_USA.csv")
        context = 40
        chunks = [df.iloc[i:i+context] for i in range(0, len(df), context)]
        maes = []
        
        for i, chunk in enumerate(chunks):
            start_idx = chunk.index[-1] + 1
            end_idx = start_idx + horizon
            if end_idx > len(df):
                print("End of data")
                break
            if len(chunk) < context:
                print("End of data")
                break
            df_horizon = df.iloc[start_idx:end_idx]
            forecast = client.forecast(chunk, h=horizon, target_col="% WEIGHTED ILI", time_col="date", model='timegpt-1')
            pred = forecast["TimeGPT"].values
            true = df_horizon["% WEIGHTED ILI"].values
            mae = (pred - true).mean()
            maes.append(mae)
            
        mae = np.mean(maes)
        print(f"Horizon {horizon}: MAE {mae}")



    