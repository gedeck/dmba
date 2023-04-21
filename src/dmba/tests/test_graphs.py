'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck
'''
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from IPython.display import Image
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from dmba import gainsChart, liftChart, textDecisionTree
from dmba.graphs import plotDecisionTree


class TestGraphs(unittest.TestCase):
    def test_liftChart(self) -> None:
        data = pd.Series([7] * 10 + [2.5] * 10 + [0.5] * 10 + [0.25] * 20 + [0.1] * 50)
        ax = liftChart(data)
        assert ax is not None

    def test_gainsChart(self) -> None:
        data = pd.Series([7] * 10 + [2.5] * 10 + [0.5] * 10 + [0.25] * 20 + [0.1] * 50)
        ax = gainsChart(data)
        assert ax is not None

    def test_textDecisionTree(self) -> None:
        iris = load_iris()
        X = iris.data
        y = iris.target
        X_train, _, y_train, _ = train_test_split(X, y, random_state=0)

        estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
        estimator.fit(X_train, y_train)

        representation = textDecisionTree(estimator)
        # print(representation)
        assert 'node=0 test node' in representation
        assert 'node=1 leaf node' in representation
        assert 'node=2 test node' in representation
        assert 'node=3 leaf node' in representation
        assert 'node=4 leaf node' in representation

    def test_plotDecisionTree(self) -> None:
        iris = load_iris()
        X = iris.data
        y = iris.target
        X_train, _, y_train, _ = train_test_split(X, y, random_state=0)

        estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
        estimator.fit(X_train, y_train)

        representation = plotDecisionTree(estimator)
        assert type(representation) == Image

        with TemporaryDirectory() as tempdir:
            pdfFile = Path(tempdir) / 'tree.pdf'
            assert not pdfFile.exists()
            representation = plotDecisionTree(estimator, pdfFile=pdfFile)
            assert pdfFile.exists()
            assert b'PDF' in pdfFile.read_bytes()
