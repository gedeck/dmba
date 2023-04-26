'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck
'''
import math
from typing import Any, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

Vector = Any


def adjusted_r2_score(y_true: Vector, y_pred: Vector, model: Any) -> float:
    """ calculate adjusted R2
    Input:
        y_true: actual values
        y_pred: predicted values
        model: predictive model
    """
    n = len(y_pred)
    p = len(model.coef_)
    if p >= n - 1:
        return 0
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def AIC_score(y_true: Vector, y_pred: Vector, model: Optional[Any] = None, df: Optional[int] = None) -> float:
    """ calculate Akaike Information Criterion (AIC)
    Input:
        y_true: actual values
        y_pred: predicted values
        model (optional): predictive model
        df (optional): degrees of freedom of model

    One of model or df is requried
    """
    p = getDegreesOfFreedom(model=model, df=df)
    n = len(y_pred)
    resid = np.array(y_true) - np.array(y_pred)
    sse = np.sum(resid ** 2)
    constant = n + n * np.log(2 * np.pi)
    return n * math.log(sse / n) + constant + 2 * (p + 1)


def getDegreesOfFreedom(model: Optional[Any] = None, df: Optional[int] = None) -> int:
    if model is not None:
        p = len(model.coef_) + 1
    elif df is not None:
        p = df
    else:
        raise ValueError('You need to provide either model or df')
    return p


def BIC_score(y_true: Vector, y_pred: Vector, model: Optional[Any] = None, df: Optional[int] = None) -> float:
    """ calculate Schwartz's Bayesian Information Criterion (AIC)
    Input:
        y_true: actual values
        y_pred: predicted values
        model: predictive model
        df (optional): degrees of freedom of model
    """
    p = getDegreesOfFreedom(model=model, df=df)
    aic = AIC_score(y_true, y_pred, model=model, df=df)
    n = len(y_pred)
    return aic - 2 * (p + 1) + math.log(n) * (p + 1)


def regressionSummary(y_true: Vector, y_pred: Vector) -> None:
    """ print regression performance metrics

    Input:
        y_true: actual values
        y_pred: predicted values
    """
    y_true = _toArray(y_true)
    y_pred = _toArray(y_pred)
    y_res = y_true - y_pred
    metrics = [
        ('Mean Error (ME)', sum(y_res) / len(y_res)),
        ('Root Mean Squared Error (RMSE)', math.sqrt(mean_squared_error(y_true, y_pred))),
        ('Mean Absolute Error (MAE)', sum(abs(y_res)) / len(y_res)),
    ]
    if all(yt != 0 for yt in y_true):
        metrics.extend([
            ('Mean Percentage Error (MPE)', 100 * sum(y_res / y_true) / len(y_res)),
            ('Mean Absolute Percentage Error (MAPE)', 100 * sum(abs(y_res / y_true) / len(y_res))),
        ])
    maxlength = max(len(m[0]) for m in metrics)
    fmt1 = f'{{:>{maxlength}}} : {{:.4f}}'
    print('\nRegression statistics\n')
    for metric, value in metrics:
        print(fmt1.format(metric, value))


def _toArray(y: Vector) -> np.ndarray:
    ya = np.asarray(y)
    if len(ya.shape) == 2 and ya.shape[1] == 1:
        ya = ya.ravel()
    return ya


def classificationSummary(y_true: Vector, y_pred: Vector, class_names: Optional[List[str]] = None) -> None:
    """ Print a summary of classification performance

    Input:
        y_true: actual values
        y_pred: predicted values
        class_names (optional): list of class names
    """
    confusionMatrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print(f'Confusion Matrix (Accuracy {accuracy:.4f})\n')

    # Pretty-print confusion matrix
    cm = confusionMatrix

    labels = class_names
    if labels is None:
        labels = [str(i) for i in range(len(cm))]

    # Convert the confusion matrix and labels to strings
    cm = [[str(i) for i in row] for row in cm]
    labels = [str(i) for i in labels]

    # Determine the width for the first label column and the individual cells
    prediction = 'Prediction'
    actual = 'Actual'
    labelWidth = max(len(s) for s in labels)
    cmWidth = max(max(len(s) for row in cm for s in row), labelWidth) + 1
    labelWidth = max(labelWidth, len(actual))

    # Construct the format statements
    fmt1 = f'{{:>{labelWidth}}}'
    fmt2 = f'{{:>{cmWidth}}}' * len(labels)

    # And print the confusion matrix
    print(fmt1.format(' ') + ' ' + prediction)
    print(fmt1.format(actual), end='')
    print(fmt2.format(*labels))

    for cls, row in zip(labels, cm):
        print(fmt1.format(cls), end='')
        print(fmt2.format(*row))
