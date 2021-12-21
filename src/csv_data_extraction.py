import pandas as pd


def extract_data_from_csv_dataset(file_path: str, output_file_path: str, column_0_name: str):
    data_from_csv = pd.read_csv(file_path, header=1)

    # extract values for column 0 and delete records without values for column 0
    column_0_values = data_from_csv.iloc[:, 0].dropna().reset_index(drop=True)

    # extract fingerprints data and delete records without values for fingerprints
    results = data_from_csv.iloc[:, 1:].dropna().reset_index(drop=True)

    # insert column 0 to fingerprints
    results.insert(0, column_0_name, column_0_values)

    # run some assertions
    assert results.columns[0] == column_0_name
    assert len(results.columns) == len(data_from_csv.columns)
    assert data_from_csv.iloc[:, 0].max() == results.iloc[:, 0].max()
    assert data_from_csv.iloc[:, 0].min() == results.iloc[:, 0].min()
    assert round(data_from_csv.iloc[:, 0].sum(), 2) == round(results.iloc[:, 0].sum(), 2)

    # save results in a csv file
    results.to_csv(output_file_path, index=False)
