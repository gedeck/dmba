# Data Mining for Business Analytics: Concepts, Techniques, and  Applications in Python

## Changelog

### 0.2 (2023-04-20)
- Move from using pydotplus to graphviz
- Add type hints
- switch to using ruff

### 0.0.19 (2021-12-05)
- Update code to latest version of packages addressing depracation warnings

### 0.0.18 (2021-02-25)
- replace 'AustralianWines.csv.gz' with correct version

### 0.0.17 (2021-02-25)
- include file 'AutoAndElectronics.zip' in package

### 0.0.16 (2021-02-25)
- expose method `get_data_file` to access files by name
```
with ZipFile(dmba.get_data_file('AutoAndElectronics.zip')) as rawData:
```

### 0.0.15 (2021-02-25)
- allow keyword arguments in `load_data`
```
import dmba
data = dmba.load_data('gdp.csv', skiprows=4)
print(data.shape)
```

### 0.0.14 (2021-02-16)
- Adapt gainsChart to work with pandas > 1.2.0 (#8)
- Make `regressionSummary` to work with column and row vectors (#6) 
- Include data sets in package, use e.g. (#10)
```
import dmba
data = dmba.load_data('Universities.csv')
print(data.shape)
```
