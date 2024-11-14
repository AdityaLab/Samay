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
    offset = -60
    # slice data in the last offset rows
    data_train = data.iloc[:offset]
    data_test = data.iloc[offset:]
    start_date = str(data_train.index[0])
    # make a list of the columns except 'ds' columns and make a list of all the time series in those columns
    column_list = data_train.columns.tolist()
    time_series_list = [np.array(data_train[column].values) for column in column_list]
    Chdataset.convert_to_arrow(data_loc, time_series=time_series_list, start_date=start_date)

    repo = "amazon/chronos-t5-small"
    # finetuning
    ch = ChronosModel(config=None, repo=repo)
    ch.finetune(training_data_paths=[data_loc], probability_list=[1])


    # log_dir = 'evaluation/finetune'
    # base_filename = 'chronos_evaluation.log'
    # logger = setup_logger(log_dir, base_filename)

    # logger.info('Started evaluating Chronos model.')
    # logger.info('Started finetuning Chronos model.')

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # # chronos_model = ChronosPipeline.from_pretrained(
    # #             "output/run-5/checkpoint-final/model_safetensors.json",
    # #             device_map=device,
    # #             torch_dtype=torch.bfloat16,
    # #         )



    # evaluation
    results = {}
    latest_run_dir = ch.get_latest_run_dir()
    model_dir = os.path.join(latest_run_dir, "checkpoint-final")
    model_type = "seq2seq"
    metrics = ['RMSE', 'MAPE']
    model = ch.load_model(model_dir, model_type)
    ch.result_logger.info(f"Model loaded from {model_dir}")
    for column_id in column_list:
        results[column_id] = ch.evaluate(data_train[column_id].values, data_test[column_id].values, abs(offset), column_id, metrics)
    
    # print(results)


if __name__ == "__main__":
    main()