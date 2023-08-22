"""
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019-2023 Galit Shmueli, Peter C. Bruce, Peter Gedeck
"""

from pathlib import Path

from polars import DataFrame, Series
import pytest

import dmba
from dmba.data import DATA_DIR


class TestData:
    def test_load_data(self):
        with pytest.raises(ValueError):
            dmba.load_data('unknown data file')

        for name in ('Amtrak.csv', ):
            data = dmba.load_data(name)
            assert isinstance(data, DataFrame)

    def test_load_data_all(self):
        for name in Path(DATA_DIR).glob('*.csv.gz'):
            data = dmba.load_data(name.name)
            assert isinstance(data, (Series, DataFrame))
            assert len(data.shape) <= 2
            if len(data.shape) == 1:
                assert isinstance(data, Series)
                print(name)
            else:
                assert isinstance(data, DataFrame)
                assert data.shape[1] > 1

    def test_kwargs_load_data(self):
        df = dmba.load_data('gdp.csv')
        org_length = len(df)
        df = dmba.load_data('gdp.csv', skip_rows=4)
        assert org_length == len(df) + 4
