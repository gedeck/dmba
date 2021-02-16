# Data Mining for Business Analytics: Concepts, Techniques, and  Applications in Python

## Changelog

### 0.0.14 (2021-02-16)
- Adapt gainsChart to work with pandas > 1.2.0 (#8)
- Make `regressionSummary` to work with column and row vectors (#6) 
- Include data sets in package, use e.g. (#10)
```
import dmba
data = dmba.load_data('Universities.csv')
print(data.shape)
```