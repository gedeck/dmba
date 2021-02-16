# Data Mining for Business Analytics: Concepts, Techniques, and  Applications in Python

## Changelog

### 0.0.14 (2021-02-16)
- Adapt gainsChart to work with pandas > 1.2.0
- Include data sets in package, use e.g.
```
import dmba
data = dmba.load_data('Universities.csv')
print(data.shape)
```