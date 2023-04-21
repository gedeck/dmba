'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck
'''
import os

import matplotlib as mpl

from .data import get_data_file, load_data
from .featureSelection import backward_elimination, exhaustive_search, forward_selection, stepwise_selection
from .graphs import gainsChart, liftChart, plotDecisionTree, textDecisionTree
from .metric import AIC_score, BIC_score, adjusted_r2_score, classificationSummary, regressionSummary
from .textMining import printTermDocumentMatrix
from .version import __version__

if os.environ.get('DISPLAY', '') == '' and os.name != 'nt':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
