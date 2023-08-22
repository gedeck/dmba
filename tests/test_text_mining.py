"""
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019-2023 Galit Shmueli, Peter C. Bruce, Peter Gedeck
"""

from contextlib import redirect_stdout
from io import StringIO

# import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from dmba import print_term_document_matrix


class TestTextMining:
    def test_print_term_document_matrix(self) -> None:
        text = ['this is the first sentence.',
                'this is a second sentence.',
                'the third sentence is here.']
        count_vect = CountVectorizer()
        counts = count_vect.fit_transform(text)

        # pd.set_option('display.width', 100)
        # pd.set_option('display.max_columns', 20)

        out = StringIO()
        with redirect_stdout(out):
            print_term_document_matrix(count_vect, counts)
        s = out.getvalue()
        assert 'S1  S2  S3' in s
        assert 'first      1   0   0' in s
        assert 'the        1   0   1' in s
