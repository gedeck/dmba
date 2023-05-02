'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019-2023 Galit Shmueli, Peter C. Bruce, Peter Gedeck
'''
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


def printTermDocumentMatrix(count_vect: CountVectorizer, counts: sp.spmatrix) -> None:
    """ Print term-document matrix created by the CountVectorizer
    Input:
        count_vect: scikit-learn Count vectorizer
        counts: term-document matrix returned by transform method of counter vectorizer
    """
    shape = counts.shape
    columns = [f'S{i}' for i in range(1, shape[0] + 1)]
    print(pd.DataFrame(data=counts.toarray().transpose(),
                       index=count_vect.get_feature_names_out(), columns=columns))
