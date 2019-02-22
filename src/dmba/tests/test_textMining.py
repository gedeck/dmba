'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and 
Applications in Python"

(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck 
'''
import unittest
from io import StringIO
from contextlib import redirect_stdout

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from dmba import printTermDocumentMatrix



class TestTextMining(unittest.TestCase):
  def test_printTermDocumentMatrix(self):
    text = ['this is the first sentence.',
            'this is a second sentence.',
            'the third sentence is here.']
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(text)

    pd.set_option('display.width', 100)
    pd.set_option('display.max_columns', 20)
    
    out = StringIO()
    with redirect_stdout(out):
      printTermDocumentMatrix(count_vect, counts)
    s = out.getvalue()
    self.assertIn('S1  S2  S3', s)
    self.assertIn('first      1   0   0', s)
    self.assertIn('the        1   0   1', s)
