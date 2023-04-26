'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck
'''
import unittest
from math import prod
from typing import Any, List

from dmba.featureSelection import Model, backward_elimination, exhaustive_search, forward_selection, stepwise_selection


class TestFeatureSelection(unittest.TestCase):
    def test_exhaustiveSearch(self) -> None:
        variables = ['a', 'b', 'c']

        def train_model(_variables: List[str]) -> Any:
            return None

        def score_model(_model: Model, variables: List[str]) -> float:
            maps = {'a': 1, 'b': 2, 'c': 3}
            return sum(maps[v] for v in variables)

        result = exhaustive_search(variables, train_model, score_model)
        assert len(result) == 3
        assert result[0]['variables'] == ['a']
        assert result[1]['variables'] == ['a', 'b']
        assert result[2]['variables'] == ['a', 'b', 'c']
        assert result[0]['score'] == 1
        assert result[1]['score'] == 3
        assert result[2]['score'] == 6

    def test_backward_elimination(self) -> None:
        variables = ['a', 'b', 'c', 'd', 'e']

        def train_model(variables: List[str]) -> Any:
            return f'Model-{"".join(variables)}'

        def score_model(_model: Model, variables: List[str]) -> float:
            maps = {'a': 4, 'b': 2, 'c': 3, 'd': -1, 'e': -2}
            return -sum(maps[v] for v in variables)

        result: Any = backward_elimination(variables, train_model, score_model, verbose=False)
        assert result[1] == ['a', 'b', 'c']

    def test_forward_elimination(self) -> None:
        variables = ['a', 'b', 'c']

        def train_model(_variables: List[str]) -> Any:
            return None

        def score_model(_model: Model, variables: List[str]) -> float:
            maps = {'a': -4, 'b': -2, 'c': 3}
            return sum(maps[v] for v in variables)

        result: Any = forward_selection(variables, train_model, score_model, verbose=False)
        assert result[1] == ['a', 'b']

    def test_stepwise_selection(self) -> None:
        variables = ['a', 'b', 'c']

        def train_model(_variables: List[str]) -> Any:
            return None

        def score_model(_model: Model, variables: List[str]) -> float:
            maps = {'a': -4, 'b': -2, 'c': 3}
            return sum(maps[v] for v in variables) - 0.1 * prod(maps[v] for v in variables)

        result: Any = stepwise_selection(variables, train_model, score_model,
                                         direction='both', verbose=False)
        assert result[1] == ['a', 'b']

        result = stepwise_selection(variables, train_model, score_model,
                                    direction='forward', verbose=False)
        assert result[1] == ['a', 'b']

        result = stepwise_selection(variables, train_model, score_model,
                                    direction='backward', verbose=False)
        assert result[1] == ['a', 'b']
