'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck
'''
import itertools
from typing import Any, Callable, Iterable, List, NamedTuple, Optional, Tuple, TypedDict, TypeVar

Model = TypeVar('Model')
TrainModel = Callable[[List[str]], Model]
ScoreModel = Callable[[Model, List[str]], float]


class ExhaustivSearchResult(TypedDict):
    n: int
    variables: List[str]
    score: float
    model: Any  # should be Model


def exhaustive_search(variables: List[str], train_model: TrainModel,
                      score_model: ScoreModel) -> List[ExhaustivSearchResult]:
    """ Variable selection using backward elimination

    Input:
        variables: complete list of variables to consider in model building
        train_model: function that returns a fitted model for a given set of variables
        score_model: function that returns the score of a model; better models have lower scores

    Returns:
        List of best subset models for increasing number of variables
    """
    # create models of increasing size and determine the best models in each case
    result = []
    for nvariables in range(1, len(variables) + 1):
        best: Optional[ExhaustivSearchResult] = None
        for varcombo in itertools.combinations(variables, nvariables):
            subset = list(varcombo)
            subset_model = train_model(subset)
            subset_score = score_model(subset_model, subset)
            if best is None or best['score'] > subset_score:
                best = ExhaustivSearchResult(
                    n=nvariables,
                    variables=subset,
                    score=subset_score,
                    model=subset_model,
                )
        assert best is not None  # noqa: ignore=S101
        result.append(best)
    return result


def backward_elimination(variables: Iterable[str], train_model: TrainModel, score_model: ScoreModel, *,
                         verbose: bool = False) -> Tuple[Model, List[str]]:
    """ Variable selection using backward elimination

    Input:
        variables: complete list of variables to consider in model building
        train_model: function that returns a fitted model for a given set of variables
        score_model: function that returns the score of a model; better models have lower scores

    Returns:
        (best_model, best_variables)
    """
    class Step(NamedTuple):
        score: float
        variable: Optional[str]
        model: Any

    # we start with a model that contains all variables
    best_variables = list(variables)
    best_model = train_model(best_variables)
    best_score = score_model(best_model, best_variables)
    if verbose:
        print('Variables: ' + ', '.join(variables))
        print(f'Start: score={best_score:.2f}')

    while len(best_variables) > 1:
        step = [Step(best_score, None, best_model)]
        for removeVar in best_variables:
            step_var = list(best_variables)
            step_var.remove(removeVar)
            step_model = train_model(step_var)
            step_score = score_model(step_model, step_var)
            step.append(Step(step_score, removeVar, step_model))

        # sort by ascending score
        step.sort(key=lambda x: x[0])

        # the first entry is the model with the lowest score
        best_score, removed_step, best_model = step[0]
        if verbose:
            print(f'Step: score={best_score:.2f}, remove {removed_step}')
        if removed_step is None:
            # step here, as removing more variables is detrimental to performance
            break
        best_variables.remove(removed_step)
    return best_model, best_variables


def forward_selection(variables: Iterable[str], train_model: TrainModel, score_model: ScoreModel, *,
                      verbose: bool = False) -> Tuple[Model, List[str]]:
    """ Variable selection using forward selection

    Input:
        variables: complete list of variables to consider in model building
        train_model: function that returns a fitted model for a given set of variables
        score_model: function that returns the score of a model; better models have lower scores

    Returns:
        (best_model, best_variables)
    """
    class Step(NamedTuple):
        score: float
        variable: Optional[str]
        model: Any

    # we start with a model that contains no variables
    best_variables: List[str] = []
    best_model = train_model(best_variables)
    best_score = score_model(best_model, best_variables)
    if verbose:
        print('Variables: ' + ', '.join(variables))
        print(f'Start: score={best_score:.2f}, constant')
    while True:
        step = [Step(best_score, None, best_model)]
        for addVar in variables:
            if addVar in best_variables:
                continue
            step_var = list(best_variables)
            step_var.append(addVar)
            step_model = train_model(step_var)
            step_score = score_model(step_model, step_var)
            step.append(Step(step_score, addVar, step_model))
        step.sort(key=lambda x: x[0])

        # the first entry in step is now the model that improved most
        best_score, added_step, best_model = step[0]
        if verbose:
            print(f'Step: score={best_score:.2f}, add {added_step}')
        if added_step is None:
            # stop here, as adding more variables is detrimental to performance
            break
        best_variables.append(added_step)
    return best_model, best_variables


def stepwise_selection(variables: List[str], train_model: TrainModel, score_model: ScoreModel, *,
                       direction: str = 'both', verbose: bool = True) -> Tuple[Model, List[str]]:
    """ Variable selection using forward and/or backward selection

    Input:
        variables: complete list of variables to consider in model building
        train_model: function that returns a fitted model for a given set of variables
        score_model: function that returns the score of a model; better models have lower scores
        direction: use it to limit stepwise selection to either 'forward' or 'backward'

    Returns:
        (best_model, best_variables)
    """
    class Step(NamedTuple):
        score: float
        variable: Optional[str]
        model: Any
        action: str

    FORWARD = 'forward'
    BACKWARD = 'backward'
    directions = [FORWARD, BACKWARD]
    if direction.lower() == FORWARD:
        directions = [FORWARD]
    if direction.lower() == BACKWARD:
        directions = [BACKWARD]

    # we start with a model that contains no variables
    best_variables: List[str] = [] if 'forward' in directions else list(variables)
    best_model = train_model(best_variables)
    best_score = score_model(best_model, best_variables)
    if verbose:
        print('Variables: ' + ', '.join(variables))
        print(f'Start: score={best_score:.2f}, constant')

    while True:
        step = [Step(best_score, None, best_model, 'unchanged')]
        if FORWARD in directions:
            for variable in variables:
                if variable in best_variables:
                    continue
                step_var = list(best_variables)
                step_var.append(variable)
                step_model = train_model(step_var)
                step_score = score_model(step_model, step_var)
                step.append(Step(step_score, variable, step_model, 'add'))

        if 'backward' in directions:
            for variable in best_variables:
                step_var = list(best_variables)
                step_var.remove(variable)
                step_model = train_model(step_var)
                step_score = score_model(step_model, step_var)
                step.append(Step(step_score, variable, step_model, 'remove'))

        # sort by ascending score
        step.sort(key=lambda x: x[0])

        # the first entry is the model with the lowest score
        best_score, chosen_variable, best_model, direction = step[0]
        if verbose:
            print(f'Step: score={best_score:.2f}, {direction} {chosen_variable}')
        if chosen_variable is None:
            # step here, as adding or removing more variables is detrimental to performance
            break
        if direction == 'add':
            best_variables.append(chosen_variable)
        else:
            best_variables.remove(chosen_variable)
    return best_model, best_variables
