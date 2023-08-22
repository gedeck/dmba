"""
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019-2023 Galit Shmueli, Peter C. Bruce, Peter Gedeck
"""

import os
import sys

import matplotlib as mpl

from .data import get_data_file, load_data
from .feature_selection import backward_elimination, exhaustive_search, forward_selection, stepwise_selection
from .graphs import gains_chart, lift_chart, plot_decision_tree, text_decision_tree
from .metric import AIC_score, BIC_score, adjusted_r2_score, classification_summary, regression_summary
from .text_mining import print_term_document_matrix


__version__ = '0.2.4'

if 'google.colab' in sys.modules:
    print('Colab environment detected.')
else:
    if os.environ.get('DISPLAY', '') == '' and os.name != 'nt':
        print('no display found. Using non-interactive Agg backend')
        mpl.use('Agg')
