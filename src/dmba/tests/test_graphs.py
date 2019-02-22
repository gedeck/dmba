'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and 
Applications in Python"

(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck, and Nitin R. Patel 
'''
import unittest

import pandas as pd

from dmba import liftChart, gainsChart

class TestGraphs(unittest.TestCase):
  def test_liftChart(self):
    data = pd.Series([7] * 10 + [2.5] * 10 + [0.5] * 10 + [0.25] * 20 + [0.1] * 50)
    ax = liftChart(data)
    self.assertIsNotNone(ax)

  def test_gainsChart(self):
    data = pd.Series([7] * 10 + [2.5] * 10 + [0.5] * 10 + [0.25] * 20 + [0.1] * 50)
    ax = gainsChart(data)
    self.assertIsNotNone(ax)
    