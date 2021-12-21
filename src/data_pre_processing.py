import pandas as pd
import numpy as np


def pre_process_data(file_path: str, output_file_path: str):
    data_train = pd.read_csv(file_path)

    # delete columns with a variance equal to zero
    data_train.drop(columns=data_train.columns[data_train.var() == 0], inplace=True)
    data_train.reset_index(drop=True, inplace=True)

    # log the values of the main parameter
    data_train[data_train.columns[0]] = np.log(data_train[data_train.columns[0]])

    # we are checking if there is no situation that in any column the minimum value == 1
    assert not any(data_train.iloc[:, 1:].min())

    # we check if the maximum value in each column == 1
    assert all(data_train.iloc[:, 1:].max())

    # let's check if the logarithmic values are correct
    test_data_train = pd.read_csv(file_path)
    assert int(data_train[data_train.columns[0]].values.sum()) == int(
        np.log(test_data_train[test_data_train.columns[0]]).sum())

    # save data after pre-processing to csv file
    data_train.to_csv(output_file_path, index=False)


file_path = "../data/pTD50_maccsfp.csv"
output_file_path = "../data/pTD50_maccsfp_after_preprocessing.csv"

pre_process_data(file_path, output_file_path)
