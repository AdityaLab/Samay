import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("data/Flu_USA/ILINet.csv")

    df = df[df["REGION TYPE"] == "National"]
    df = df[df["WEEK"] != 53]

    df["date"] = pd.to_datetime(
        df["YEAR"].astype(str) + "-W" + df["WEEK"].astype(str) + "-1",
        format="%Y-W%U-%w",
    )

    result = []
    for i in range(len(df) - 1):
        result.append(df.iloc[i])

        if (df.iloc[i + 1]["date"] - df.iloc[i]["date"]).days == 14:
            new_row = df.iloc[i].copy()
            new_row["date"] = df.iloc[i]["date"] + pd.Timedelta(days=7)
            result.append(new_row)

    result.append(df.iloc[-1])
    df = pd.DataFrame(result)

    df = df.drop(columns=["YEAR", "WEEK"])
    gaps = df["date"].diff().dropna().unique()
    print("Unique time intervals:", gaps)
    df["time_diff"] = df["date"].diff()

    rows_with_14_days = df[df["time_diff"] == pd.Timedelta(days=14)]
    print(rows_with_14_days)
    df = df.drop(columns=["time_diff"])
    infered_freq = pd.infer_freq(df["date"])
    print(f"Infered frequency: {infered_freq}")

    df.to_csv("data/Flu_USA/Flu_USA.csv", index=False)

    print("Data saved to output.csv")
