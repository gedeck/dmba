'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and
Applications in Python"

(c) 2019-2023 Galit Shmueli, Peter C. Bruce, Peter Gedeck
'''
from pathlib import Path

import setuptools


def getVersion():
    f = Path(__file__).parent / 'src' / 'dmba' / 'version.py'
    lines = f.read_text().split('\n')
    version = [s for s in lines if '__version__' in s][0]
    version = version.split('=')[1].strip().strip("'")
    return version

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
