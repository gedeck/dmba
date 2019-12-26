'''
Utility functions for "Data Mining for Business Analytics: Concepts, Techniques, and 
Applications in Python"

(c) 2019 Galit Shmueli, Peter C. Bruce, Peter Gedeck 
'''
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dmba",
    version="0.0.11",
    author="Peter Gedeck",
    author_email="mail@petergedeck.com",
    description="Utility functions for 'Data Mining for Business Analytics: Concepts, Techniques, and Applications in Python'",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gedeck/dmba",
    packages=setuptools.find_packages("src"),
    package_dir={'':'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    
    test_suite='nose.collector',
    tests_require=['nose'],
)
