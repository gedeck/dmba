'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and 
Applications in Python"

(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck
'''
import unittest

import pandas as pd

from dmba import liftChart, gainsChart, textDecisionTree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class TestGraphs(unittest.TestCase):
  def test_liftChart(self):
    data = pd.Series([7] * 10 + [2.5] * 10 + [0.5] * 10 + [0.25] * 20 + [0.1] * 50)
    ax = liftChart(data)
    self.assertIsNotNone(ax)

  def test_gainsChart(self):
    data = pd.Series([7] * 10 + [2.5] * 10 + [0.5] * 10 + [0.25] * 20 + [0.1] * 50)
    ax = gainsChart(data)
    self.assertIsNotNone(ax)

  def test_textDecisionTree(self):
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
    estimator.fit(X_train, y_train)

    representation = textDecisionTree(estimator)
    # print(representation)
    self.assertIn('node=0 test node', representation)
    self.assertIn('node=1 leaf node', representation)
    self.assertIn('node=2 test node', representation)
    self.assertIn('node=3 leaf node', representation)
    self.assertIn('node=4 leaf node', representation)
