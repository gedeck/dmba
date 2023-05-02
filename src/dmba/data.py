'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019-2023 Galit Shmueli, Peter C. Bruce, Peter Gedeck
'''
from pathlib import Path
from typing import Any, Union

import pandas as pd

DATA_DIR = Path(__file__).parent / 'csvFiles'


def load_data(name: str, **kwargs: Any) -> Union[pd.DataFrame, pd.Series]:
    """ Returns the data either as a Pandas data frame or series """
    data_file = get_data_file(name)
    if not data_file.exists():
        raise ValueError('Data file {name} not found')
    data = pd.read_csv(data_file, **kwargs)
    if data.shape[1] == 1:
        return data[data.columns[0]]  # pylint: disable=E1136
    return data


def get_data_file(name: str) -> Path:
    if name.endswith('.zip'):
        return DATA_DIR / name
    if name.endswith('.gz'):
        name = name[:-3]
    if name.endswith('.csv'):
        name = name[:-4]
    return DATA_DIR / f'{name}.csv.gz'
