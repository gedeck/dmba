"""
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019-2023 Galit Shmueli, Peter C. Bruce, Peter Gedeck
"""

import gzip
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import polars as pl
from polars import DataFrame, Series


DATA_DIR = Path(__file__).parent.parent / 'data'


def load_data(name: str, **kwargs: Any) -> DataFrame | Series:
    """ Returns the data either as a Pandas data frame or series """
    data_path = DATA_DIR / name
    if not data_path.exists():
        raise ValueError(f'Data file {data_path} not found')
    if data_path.suffixes == ['.csv']:
        data = pl.read_csv(data_path, **kwargs)
    else:
        data = pl.read_csv(get_data(data_path), **kwargs)
    if data.shape[1] == 1:
        return data[data.columns[0]]  # pylint: disable=E1136
    return data

def get_data(path: Path) -> bytes:
    """Returns the data as a byte string"""
    if path.suffix == '.zip':
        with (
            ZipFile(path) as zip_file,
            zip_file.open(zip_file.namelist()[0]) as data_file,
        ):
            return data_file.read()
    elif path.suffixes == ['.csv', '.gz']:
        with gzip.open(path) as data_file:
            return data_file.read()
    else:
        raise ValueError('Path with unknown suffixes: {path}')
