from src.tsfmproject.model import ChronosModel
from src.tsfmproject.dataset import ChronosDataset
import torch
import numpy as np
import os


def main():
    Chdataset = ChronosDataset(name="tycho", path='data/dataset/timesfm_covid_pivot.csv')
    Chdataset.process_covid_data()
    data = Chdataset.dataset
    data_loc = './src/tsfmproject/models/chronosforecasting/data/us_data.arrow'
    initial_train_length = 150
    evaluation_length = 60
    increment_length = 30

    repo = "amazon/chronos-t5-small"
    ch = ChronosModel(config=None, repo=repo)

    results = {}

    train_length = initial_train_length
    while train_length + evaluation_length <= len(data):
        data_train = data.iloc[:train_length]
        data_test = data.iloc[train_length:train_length + evaluation_length]

        if data_train.empty or data_test.empty:
            print(f"Skipping iteration with train_length={train_length} due to empty dataset.")
            train_length += increment_length
            continue

        start_date = str(data_train.index[0])
        column_list = data_train.columns.tolist()
        time_series_list = [np.array(data_train[column].values) for column in column_list]
        Chdataset.convert_to_arrow(data_loc, time_series=time_series_list, start_date=start_date)

        # Finetuning
        print(f"Finetuning with train_length={train_length}")
        ch.finetune(training_data_paths=[data_loc], probability_list=[1])

        # Evaluation
        latest_run_dir = ch.get_latest_run_dir()
        model_dir = os.path.join(latest_run_dir, "checkpoint-final")
        model_type = "seq2seq"
        metrics = ['RMSE', 'MAPE']
        ch.load_model(model_dir, model_type)
        ch.result_logger.info(f"Model loaded from {model_dir}")
        print(f"Total columns: {len(column_list)}")

        for column_id in column_list:
            results[column_id] = ch.evaluate(data_train[column_id].values, data_test[column_id].values, evaluation_length, column_id, metrics)

        # Increment the training length
        train_length += increment_length

    # Print or save the results
    # print(results)


if __name__ == "__main__":
    main()