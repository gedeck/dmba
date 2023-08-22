"""
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019-2023 Galit Shmueli, Peter C. Bruce, Peter Gedeck
"""

import setuptools


setuptools.setup(
    install_requires=[
        'graphviz',
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
    ],
    extras_require={
        "ipython": ["ipython"],
    },
    test_suite='nose.collector',
    tests_require=['nose'],
)
