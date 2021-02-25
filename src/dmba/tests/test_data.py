'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and 
Applications in Python"

(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck 
'''
from pathlib import Path
import unittest

import pytest

from dmba.data import DATA_DIR
import dmba
import pandas as pd


class TestData(unittest.TestCase):
    def test_load_data(self):
        with pytest.raises(ValueError):
            dmba.load_data('unknown data file')

        for name in ('Amtrak.csv', ):
            data = dmba.load_data(name)
            assert isinstance(data, pd.DataFrame)

    def test_load_data_all(self):
        for name in Path(DATA_DIR).glob('*.csv.gz'):
            data = dmba.load_data(name.name)
            assert isinstance(data, (pd.Series, pd.DataFrame))
            assert len(data.shape) <= 2
            if len(data.shape) == 1:
                assert isinstance(data, pd.Series)
                print(name)
            else:
                assert isinstance(data, pd.DataFrame)
                assert data.shape[1] > 1

    def test_kwargs_load_data(self):
        df = dmba.load_data('gdp.csv')
        org_length = len(df)
        df = dmba.load_data('gdp.csv', skiprows=4)
        assert org_length == len(df) + 4

    def test_get_data_file(self):
        assert dmba.get_data_file('AutoAndElectronics.zip').exists()
        assert dmba.get_data_file('gdp.csv').exists()
        assert dmba.get_data_file('gdp.csv.gz').exists()
