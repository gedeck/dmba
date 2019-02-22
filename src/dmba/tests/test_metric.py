'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and 
Applications in Python"

(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck, and Nitin R. Patel 
'''
import unittest
from collections import namedtuple
from sklearn.metrics.regression import r2_score
from dmba.metric import adjusted_r2_score, AIC_score, BIC_score


MockModel = namedtuple('Model', 'coef_')

class TestMetric(unittest.TestCase):
  def test_adjusted_r2_score(self):
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1, 3, 2, 5, 4]
    
    r2 = r2_score(y_true, y_pred)
    assert r2 == 0.6
    
    for df in range(4):
      n = len(y_true)
      f = (n - 1) / (n - df - 1)
      expected = 1 - (1 - r2) * f
      self.assertAlmostEqual(
        adjusted_r2_score(y_true, y_pred, MockModel(coef_=[1] * df)),
        expected, places=3, msg=f'failed for df={df}')

    # if degree of freedom gets too large returns 0
    coef = [1] * 4
    self.assertAlmostEqual(
      adjusted_r2_score(y_true, y_pred, MockModel(coef_=coef)),
      0, places=3, msg=f'failed for df={df}')

    coef = [1] * 5
    self.assertAlmostEqual(
      adjusted_r2_score(y_true, y_pred, MockModel(coef_=coef)),
      0, places=3, msg=f'failed for df={df}')
    
  def test_AIC_score(self):
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1, 3, 2, 5, 4]
    
    self.assertAlmostEqual(
      AIC_score(y_true, y_pred, MockModel(coef_=[1] * 2)),
      21.0736, places=3)
    
    self.assertAlmostEqual(
      AIC_score(y_true, y_pred, df=3),
      AIC_score(y_true, y_pred, MockModel(coef_=[1] * 2)), places=3)

    self.assertGreater(
      AIC_score(y_true, y_pred, df=3),
      AIC_score(y_true, y_pred, df=2))

  def test_BIC_score(self):
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1, 3, 2, 5, 4]
    
    self.assertAlmostEqual(
      BIC_score(y_true, y_pred, MockModel(coef_=[1] * 2)),
      19.51141, places=3)
    
    self.assertAlmostEqual(
      BIC_score(y_true, y_pred, df=3),
      BIC_score(y_true, y_pred, MockModel(coef_=[1] * 2)), places=3)

    self.assertGreater(
      BIC_score(y_true, y_pred, df=3),
      BIC_score(y_true, y_pred, df=2))
