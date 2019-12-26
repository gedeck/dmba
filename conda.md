```
conda create --name dmba python=3.7
conda activate dmba
conda install matplotlib
conda install pandas
conda install scikit-learn

conda install tox
conda install twine
```

```
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*
```