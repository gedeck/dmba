"""
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019-2023 Galit Shmueli, Peter C. Bruce, Peter Gedeck
"""

import gzip
from pathlib import Path
from typing import Any, Union
from zipfile import ZipFile

import polars as pl
from polars import DataFrame, Series


DATA_DIR = Path(__file__).parent.parent / 'data'


def load_data(name: str, **kwargs: Any) -> Union[DataFrame, Series]:
    """ Returns the data either as a Pandas data frame or series """
    data = pl.read_csv(get_bytes(name), **kwargs)
    if data.shape[1] == 1:
        return data[data.columns[0]]  # pylint: disable=E1136
    return data

def get_bytes(name: str) -> bytes:
    """Returns the data as a byte string"""
    data_file = get_data_file(name)
    if not data_file.exists():
        raise ValueError('Data file {name} not found')
    if data_file.suffix == '.zip':
        with ZipFile(data_file) as zip_file:
            with zip_file.open(zip_file.namelist()[0]) as data_file:
                return data_file.read()
    elif data_file.suffixes == ['.csv', '.gz']:
        with gzip.open(data_file) as data_file:
            return data_file.read()

def get_data_file(name: str) -> Path:
    stem, *suffixes = name.split('.')
    match suffixes:
        case ['.zip']:
            return DATA_DIR / name
        case ['.gz'] | ['.csv']:
            return DATA_DIR / stem / '.csv.gz'
