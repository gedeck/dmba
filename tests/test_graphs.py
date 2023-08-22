"""
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019-2023 Galit Shmueli, Peter C. Bruce, Peter Gedeck
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from polars import Series
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from dmba import gains_chart, lift_chart, text_decision_tree
from dmba.graphs import plot_decision_tree

try:
    from IPython.display import Image
    HAS_IMAGE = True
except ImportError:
    HAS_IMAGE = False


class TestGraphs:
    # def test_lift_chart(self) -> None:
    #     data = Series([7] * 10 + [2.5] * 10 + [0.5] * 10 + [0.25] * 20 + [0.1] * 50)
    #     ax = lift_chart(data)
    #     assert ax is not None

    # def test_gains_chart(self) -> None:
    #     data = Series([7] * 10 + [2.5] * 10 + [0.5] * 10 + [0.25] * 20 + [0.1] * 50)
    #     ax = gains_chart(data)
    #     assert ax is not None

    def test_text_decision_tree(self) -> None:
        iris = load_iris()
        X = iris.data
        y = iris.target
        X_train, _, y_train, _ = train_test_split(X, y, random_state=0)

        estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
        estimator.fit(X_train, y_train)

        representation = text_decision_tree(estimator)
        # print(representation)
        assert 'node=0 test node' in representation
        assert 'node=1 leaf node' in representation
        assert 'node=2 test node' in representation
        assert 'node=3 leaf node' in representation
        assert 'node=4 leaf node' in representation

    def test_plot_decision_tree(self) -> None:
        iris = load_iris()
        X = iris.data
        y = iris.target
        X_train, _, y_train, _ = train_test_split(X, y, random_state=0)

        estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
        estimator.fit(X_train, y_train)

        representation = plot_decision_tree(estimator)
        if HAS_IMAGE:
            assert type(representation) == Image
        else:
            assert 'You need to install Image and/or graphviz' in representation

        with TemporaryDirectory() as tempdir:
            pdf_file = Path(tempdir) / 'tree.pdf'
            assert not pdf_file.exists()
            representation = plot_decision_tree(estimator, pdf_file=pdf_file)
            assert pdf_file.exists()
            assert b'PDF' in pdf_file.read_bytes()
