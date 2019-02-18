'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and 
Applications in Python"

(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck, and Nitin R. Patel 
'''
import math
import numpy as np
from sklearn.metrics import classification, regression


def regressionSummary(y_true, y_pred):
    """ print regression performance metrics 
    
    Input:
        y_true: actual values
        y_pred: predicted values
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_res = y_true - y_pred
    metrics = [
        ('Mean Error (ME)', sum(y_res) / len(y_res)),
        ('Root Mean Squared Error (RMSE)', math.sqrt(regression.mean_squared_error(y_true, y_pred))),
        ('Mean Absolute Error (MAE)', sum(abs(y_res)) / len(y_res)),
        ('Mean Percentage Error (MPE)', 100 * sum(y_res / y_true) / len(y_res)),
        ('Mean Absolute Percentage Error (MAPE)', 100 * sum(abs(y_res / y_true) / len(y_res))),
    ]
    fmt1 = '{{:>{}}} : {{:.4f}}'.format(max(len(m[0]) for m in metrics))
    print('\nRegression statistics\n')
    for metric, value in metrics:
        print(fmt1.format(metric, value))


def classificationSummary(y_true, y_pred, class_names=None):
    """ Print a summary of classification performance
    
    Input:
        y_true: actual values
        y_pred: predicted values
        class_names (optional): list of class names
    """
    confusionMatrix = classification.confusion_matrix(y_true, y_pred)
    accuracy = classification.accuracy_score(y_true, y_pred)

    print('Confusion Matrix (Accuracy {:.4f})\n'.format(accuracy))
    
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
    fmt1 = '{{:>{}}}'.format(labelWidth)
    fmt2 = '{{:>{}}}'.format(cmWidth) * len(labels)

    # And print the confusion matrix    
    print(fmt1.format(' ') + ' ' + prediction)
    print(fmt1.format(actual), end='')
    print(fmt2.format(*labels))
    
    for cls, row in zip(labels, cm):
        print(fmt1.format(cls), end='')
        print(fmt2.format(*row))
