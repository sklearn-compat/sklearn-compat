## Ease multi-version support for scikit-learn compatible library

[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

`sklearn_compat` is a small Python package that allows you to support new
scikit-learn features with older versions of scikit-learn.

The aim is to support a range of scikit-learn versions as specified in the
[SPEC0](https://scientific-python.org/specs/spec-0000/). It means that you will find
utility to support the last 4 released version of scikit-learn.

## How to adapt your scikit-learn code

### Upgrading to scikit-learn 1.6

#### `validate_data`

Your previous code could have looked like this:

```python
class MyEstimator(BaseEstimator):
    def fit(self, X, y=None):
        X = self._validate_data(X, force_all_finite=True)
        return self
```

There is two major changes in scikit-learn 1.6:

- `validate_data` has been moved to `sklearn.utils.validation`.
- `force_all_finite` is deprecated in favor of the `ensure_all_finite` parameter.

You can now use the following code for backward compatibility:

```python
from sklearn_compat.utils.validation import validate_data

class MyEstimator(BaseEstimator):
    def fit(self, X, y=None):
        X = validate_data(self, X=X, ensure_all_finite=True)
        return self
```
