from datasets import load_dataset
import os


if __name__ == "__main__":
    save_dir = "data/monash"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataset_names = [
    # "weather",
    # "tourism_yearly",
    # "tourism_quarterly",
    # "tourism_monthly",
    # "cif_2016",
    # "london_smart_meters",
    # "australian_electricity_demand",
    # "wind_farms_minutely",
    # "bitcoin",
    # "pedestrian_counts",
    # "vehicle_trips",
    # "kdd_cup_2018",
    # "nn5_daily",
    # "nn5_weekly",
    # "kaggle_web_traffic",
    # "kaggle_web_traffic_weekly",
    # "solar_10_minutes",
    # "solar_weekly",
    # "car_parts",
    # "fred_md",
    # "traffic_hourly",
    # "traffic_weekly",
    # "hospital",
    # "covid_deaths",
    # "sunspot",
    # "saugeenday",
    # "us_births",
    # "solar_4_seconds",
    # "wind_4_seconds",
    # "rideshare",
    "oikolab_weather",
    "temperature_rain"
]
    for dataset_name in dataset_names:
        dataset = load_dataset("monash_tsf", dataset_name)
        dataset.save_to_disk(f"{save_dir}/{dataset_name}")
        print(f"Downloaded {dataset_name} dataset")