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

#### `_check_n_features` and `_check_feature_names`

Similarly to `validate_data`, these two functions have been moved to
`sklearn.utils.validation` instead of being methods of the estimators. So the following
code:

```python
class MyEstimator(BaseEstimator):
    def fit(self, X, y=None):
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)
        return self
```

becomes:

```python
from sklearn_compat.utils.validation import _check_n_features, _check_feature_names

class MyEstimator(BaseEstimator):
    def fit(self, X, y=None):
        _check_n_features(self, X, reset=True)
        _check_feature_names(self, X, reset=True)
        return self
```

### Upgrading to scikit-learn 1.5

#### `_get_column_indices`

The utility function `_get_column_indices` has been moved from `sklearn.utils` to
`sklearn.utils._indexing`.

So the following code:

```python
import pandas as pd
from sklearn.utils import _get_column_indices

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
_get_column_indices(df, key="b")
```

becomes:

```python
import pandas as pd
from sklearn_compat.utils._indexing import _get_column_indices

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
_get_column_indices(df, key="b")
```

#### `_print_elapsed_time`

The function `_print_elapsed_time` has been moved from `sklearn.utils` to
`sklearn.utils._user_interface`.

So the following code:

```python
from sklearn.utils import _print_elapsed_time

with _print_elapsed_time("sklearn_compat", "testing"):
    time.sleep(0.1)
```

becomes:

```python
from sklearn_compat.utils._user_interface import _print_elapsed_time

with _print_elapsed_time("sklearn_compat", "testing"):
    time.sleep(0.1)
```

### Upgrading to scikit-learn 1.4

### Support for metadata routing

TODO

### Upgrading to scikit-learn 1.3

#### Parameter validation

scikit-learn 1.3 introduced a new way to validate the parameters is a consistent manner.
One could in the past define the following class:

```python
class MyEstimator(BaseEstimator):
    def __init__(self, a=1):
        self.a = a

    def fit(self, X, y=None):
        if self.a < 0:
            raise ValueError("a must be positive")
        return self
```

becomes:

```python
from sklearn_compat.base import ParamsValidationMixin

class MyEstimator(ParamsValidationMixin, BaseEstimator):
    _parameter_constraints = {"a": [Interval(Integral, 0, None, closed="left")]}

    def __init__(self, a=1):
        self.a = a

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        return self
```

The advantage is that the error raised will be more informative and consistent across
estimators. Also, we have the possibility to skip the validation of the parameters when
using this estimator as a meta-estimator.
