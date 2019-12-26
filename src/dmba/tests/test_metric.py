'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and 
Applications in Python"

(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck 
'''
import unittest
from collections import namedtuple
from contextlib import redirect_stdout
from io import StringIO

from sklearn.metrics import r2_score

from dmba import adjusted_r2_score, AIC_score, BIC_score
from dmba import regressionSummary, classificationSummary


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

  def test_regressionSummary(self):
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1, 3, 2, 5, 4]

    out = StringIO()
    with redirect_stdout(out):
      regressionSummary(y_true, y_pred)
    s = out.getvalue()
    self.assertIn('Regression statistics', s)
    self.assertIn('(ME) : 0.0000', s)
    self.assertIn('(RMSE) : 0.8944', s)
    self.assertIn('(MAE) : 0.8000', s)
    self.assertIn('(MPE) : -4.3333', s)
    self.assertIn('(MAPE) : 25.6667', s)

    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 2, 1, 4, 3]

    out = StringIO()
    with redirect_stdout(out):
      regressionSummary(y_true, y_pred)
    s = out.getvalue()
    self.assertIn('Regression statistics', s)
    self.assertIn('(ME) : 0.0000', s)
    self.assertIn('(RMSE) : 0.8944', s)
    self.assertIn('(MAE) : 0.8000', s)
    self.assertNotIn('(MPE)', s)
    self.assertNotIn('(MAPE)', s)

  def test_classificationSummary(self):
    y_true = [1, 0, 0, 1, 1, 1]
    y_pred = [1, 0, 1, 1, 0, 0]

    out = StringIO()
    with redirect_stdout(out):
      classificationSummary(y_true, y_pred, class_names=['a', 'b'])
    s = out.getvalue()

    self.assertIn('Confusion Matrix', s)
    self.assertIn('       Prediction', s)
    self.assertIn('a 1 1', s)
    self.assertIn('b 2 2', s)
