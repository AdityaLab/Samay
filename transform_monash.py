from samay.utils import arrow_to_csv
import os
import pandas as pd

FREQS = {
    "weather": "1D",
    "tourism_yearly": "1YE",
    "tourism_quarterly": "1Q",
    "tourism_monthly": "1M",
    "cif_2016": "1M",
    "london_smart_meters": "30min",
    "australian_electricity_demand": "30min",
    "wind_farms_minutely": "1min",
    "bitcoin": "1D",
    "pedestrian_counts": "1h",
    "vehicle_trips": "1D",
    "kdd_cup_2018": "1H",
    "nn5_daily": "1D",
    "nn5_weekly": "1W",
    "kaggle_web_traffic": "1D",
    "kaggle_web_traffic_weekly": "1W",
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
    "temperature_rain": "1D"
}


if __name__ == "__main__":
    monash_dir = "data/monash"
    dataset_list = os.listdir(monash_dir)
    splits = ["train", "validation", "test"]
    for dataset in dataset_list:
        # if not dataset in ["rideshare"]:
        #     continue
        print(f"Converting {dataset} dataset")
        for split in splits:
            arrow_dir = os.path.join(monash_dir, dataset, split)
            freq = FREQS[dataset]
            arrow_to_csv(arrow_dir, freq)
            csv_file = os.path.join(monash_dir, dataset, split + "/data.csv")
            df = pd.read_csv(csv_file)
            # fill missing values with 0
            df.fillna(0, inplace=True)
            df.to_csv(csv_file, index=False)
            





            
