from tsfmproject.model import TimesfmModel, TimesfmDataset, ChronosModel, ChronosDataset
import torch
import numpy as np


def main():
    Chdataset = ChronosDataset(name="tycho", path='data/datasets/timesfm_covid_pivot.csv')
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
    
    ch = ChronosModel(repo=repo)
    ch.finetune(training_data_paths=data_loc, probability_list=[1])


if __name__ == "__main__":
    main()