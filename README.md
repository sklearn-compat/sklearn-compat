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

In scikit-learn 1.5, many developer utilities have been moved to dedicated modules.
We provide a compatibility layer such that you don't have to check the version or try
to import the utilities from different modules.

In the future, when supporting scikit-learn 1.6+, you will have to change the import
from:

```python
from sklearn_compat.utils._indexing import _safe_indexing
```

to

```python
from sklearn.utils._indexing import _safe_indexing
```

Thus, the module path will already be correct. Now, we will go into details for
each module and function impacted.

#### `extmath` module

The function `safe_sqr` and `_approximate_mode` have been moved from `sklearn.utils` to
`sklearn.utils.extmath`.

So some code looking like this:

```python
from sklearn.utils import safe_sqr, _approximate_mode

safe_sqr(np.array([1, 2, 3]))
_approximate_mode(class_counts=np.array([4, 2]), n_draws=3, rng=0)
```

becomes:

```python
from sklearn_compat.utils.extmath import safe_sqr, _approximate_mode

safe_sqr(np.array([1, 2, 3]))
_approximate_mode(class_counts=np.array([4, 2]), n_draws=3, rng=0)
```

#### `_indexing`

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

#### `_user_interface` module

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

#### `_optional_dependencies` module

The functions `check_matplotlib_support` and `check_pandas_support` have been moved from
`sklearn.utils` to `sklearn.utils._optional_dependencies`.

So the following code:

```python
from sklearn.utils import check_matplotlib_support, check_pandas_support

check_matplotlib_support("sklearn_compat")
check_pandas_support("sklearn_compat")
```

becomes:

```python
from sklearn_compat.utils._optional_dependencies import (
    check_matplotlib_support, check_pandas_support
)

check_matplotlib_support("sklearn_compat")
check_pandas_support("sklearn_compat")
```

### Upgrading to scikit-learn 1.4

#### `process_routing`

The signature of the `process_routing` function changed in scikit-learn 1.4. You don't
need to change the import but only the call to the function. Calling the function with
the new signature will be compatible with the previous signature as well. So a code
looking like this:

```python
class MetaEstimator(BaseEstimator):
    def fit(self, X, y, sample_weight=None, **fit_params):
        params = process_routing(self, "fit", fit_params, sample_weight=sample_weight)
        return self
```

becomes:

```python
class MetaEstimator(BaseEstimator):
    def fit(self, X, y, sample_weight=None, **fit_params):
        params = process_routing(self, "fit", sample_weight=sample_weight, **fit_params)
        return self
```

### Parameter validation

scikit-learn introduced a new way to validate parameters at `fit` time. The recommended
way to support this feature in scikit-learn 1.3+ is to inherit from
`sklearn.base.BaseEstimator` and decorate the `fit` method using the decorator
`sklearn.base._fit_context`. In this package, we provide a mixin class in case you
don't want to inherit from `sklearn.base.BaseEstimator`.

So a small example could have been in the past:

```python
class MyEstimator:
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

class MyEstimator(ParamsValidationMixin):
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
