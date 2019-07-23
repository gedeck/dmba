'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and 
Applications in Python"

(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck 
'''
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

from .featureSelection import exhaustive_search, forward_selection, backward_elimination, stepwise_selection
from .graphs import plotDecisionTree, liftChart, gainsChart, textDecisionTree
from .metric import regressionSummary, classificationSummary
from .metric import AIC_score, BIC_score, adjusted_r2_score
from .textMining import printTermDocumentMatrix
