Create environment
```
conda create --name dmba python=3.7
conda activate dmba
pip install matplotlib pandas scikit-learn tox twine
```
Testing
```
tox
```
Distribute
```
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*
```