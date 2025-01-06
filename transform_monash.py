from src.tsfmproject.utils import arrow_to_csv
import os
import pandas as pd


if __name__ == "__main__":
    monash_dir = "data/monash"
    dataset_list = os.listdir(monash_dir)
    splits = ["train", "validation", "test"]
    for dataset in dataset_list:
        if not dataset in ["oikolab_weather", "temperature_rain"]:
            continue
        print(f"Converting {dataset} dataset")
        for split in splits:
            arrow_dir = os.path.join(monash_dir, dataset, split)
            arrow_to_csv(arrow_dir)
            csv_file = os.path.join(monash_dir, dataset, split + "/data.csv")
            df = pd.read_csv(csv_file)
            # fill missing values with 0
            df.fillna(0, inplace=True)
            df.to_csv(csv_file, index=False)
            





            
