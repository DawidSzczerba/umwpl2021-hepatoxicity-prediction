import pandas as pd


def prepare_data(file_path: str, output_file_path: str, column_0_name: str):
    data = pd.read_csv(file_path, header=1)
    data_values = data.iloc[::2, 0].reset_index(drop=True)
    finger_prints = data.iloc[1::2, ].reset_index(drop=True).drop('Unnamed: 0', axis=1)
    finger_prints.insert(0, column_0_name, data_values)
    finger_prints.to_csv(output_file_path, index=False)
