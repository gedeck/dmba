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


long_description = Path("README.md").read_text()

setuptools.setup(
    name="dmba",
    version=getVersion(),
    author="Peter Gedeck",
    author_email="mail@petergedeck.com",
    description=("Utility functions for 'Data Mining for Business Analytics: "
                 "Concepts, Techniques, and Applications in Python'"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gedeck/dmba",
    packages=setuptools.find_packages("src"),
    package_dir={'': 'src'},
    package_data={
        "dmba": ["csvFiles/*.csv.gz", "csvFiles/*.zip"],
    },
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
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    test_suite='nose.collector',
    tests_require=['nose'],
)
