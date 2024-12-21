# Ease multi-version support for scikit-learn compatible library

[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)
![GitHub Actions CI](https://github.com/sklearn-compat/sklearn-compat/actions/workflows/testing.yml/badge.svg)
[![codecov](https://codecov.io/gh/sklearn-compat/sklearn-compat/graph/badge.svg?token=cndAFPqxhF)](https://codecov.io/gh/sklearn-compat/sklearn-compat)
[![Python Version](https://img.shields.io/pypi/pyversions/sklearn-compat.svg)](https://img.shields.io/pypi/pyversions/sklearn-compat.svg)
[![PyPI](https://badge.fury.io/py/sklearn-compat.svg)](https://badge.fury.io/py/sklearn-compat.svg)

`sklearn-compat` is a small Python package that help developer writing scikit-learn
compatible estimators to support multiple scikit-learn versions. Note that we provide
a vendorable version of this package in the `src/sklearn_compat/_sklearn_compat.py`
file if you do not want to depend on `sklearn-compat` as a package. The package is
available on [PyPI](https://pypi.org/project/sklearn-compat/) and on
[conda-forge](https://anaconda.org/conda-forge/sklearn-compat).

As maintainers of third-party libraries depending on scikit-learn such as
[`imbalanced-learn`](https://github.com/scikit-learn-contrib/imbalanced-learn),
[`skrub`](https://github.com/skrub-data/skrub), or
[`skops`](https://github.com/skops-dev/skops), we usually identified small breaking
changes on the "private" developer utilities of `scikit-learn`. Indeed, each of these
third-party libraries code the exact same utilities when it comes to support multiple
`scikit-learn` versions. We therefore decided to factorize these utilities in a
dedicated package that we update at each `scikit-learn` release.

When it comes to support multiple `scikit-learn` versions, the initial plan as of
December 2024 is to follow the [SPEC0](https://scientific-python.org/specs/spec-0000/)
recommendations. It means that this utility will support **at least** the `scikit-learn`
versions up to 2 years or about 4 versions. The current version of `sklearn-compat`
supports `scikit-learn` >= 1.2.

## How to adapt your scikit-learn code

In this section, we describe succinctly the changes you need to do to your code to
support multiple `scikit-learn` versions using `sklearn-compat` as a package. If you
use the vendored version of `sklearn-compat`, all imports will be changed from:

``` py
from sklearn_compat.any_submodule import any_function
```

to

``` py
from path.to._sklearn_compat import any_function
```

where `_sklearn_compat` is the vendored version of `sklearn-compat` in your project.

### Upgrading to scikit-learn 1.6

#### `is_clusterer` function

The function `is_clusterer` has been added in `scikit-learn` 1.6. So we backport it
such that you can have access to it in scikit-learn 1.2+. The pattern is the following:

``` py
from sklearn.cluster import KMeans
from sklearn_compat.base import is_clusterer

is_clusterer(KMeans())`
```

#### `validate_data` function

Your previous code could have looked like this:

``` py
class MyEstimator(BaseEstimator):
    def fit(self, X, y=None):
        X = self._validate_data(X, force_all_finite=True)
        return self
```

There is two major changes in scikit-learn 1.6:

- `validate_data` has been moved to `sklearn.utils.validation`.
- `force_all_finite` is deprecated in favor of the `ensure_all_finite` parameter.

You can now use the following code for backward compatibility:

``` py
from sklearn_compat.utils.validation import validate_data

class MyEstimator(BaseEstimator):
    def fit(self, X, y=None):
        X = validate_data(self, X=X, ensure_all_finite=True)
        return self
```

#### `check_array` and `check_X_y` functions

The parameter `force_all_finite` has been deprecated in favor of the `ensure_all_finite`
parameter. You need to modify the call to the function to use the new parameter. So,
the change is the same as for `validate_data` and will look like this:

``` py
from sklearn.utils.validation import check_array, check_X_y

check_array(X, force_all_finite=True)
check_X_y(X, y, force_all_finite=True)
```

to:

``` py
from sklearn_compat.utils.validation import check_array, check_X_y

check_array(X, ensure_all_finite=True)
check_X_y(X, y, ensure_all_finite=True)
```

#### `_check_n_features` and `_check_feature_names` functions

Similarly to `validate_data`, these two functions have been moved to
`sklearn.utils.validation` instead of being methods of the estimators. So the following
code:

``` py
class MyEstimator(BaseEstimator):
    def fit(self, X, y=None):
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)
        return self
```

becomes:

``` py
from sklearn_compat.utils.validation import _check_n_features, _check_feature_names

class MyEstimator(BaseEstimator):
    def fit(self, X, y=None):
        _check_n_features(self, X, reset=True)
        _check_feature_names(self, X, reset=True)
        return self
```

Note that it is best to call `validate_data` with `skip_check_array=True` instead of
calling these private functions. See the section above regarding `validate_data`.

#### `Tags`, `__sklearn_tags__` and estimator tags

The estimator tags infrastructure in scikit-learn 1.6 has changed. In order to be
compatible with multiple scikit-learn versions, your estimator should implement both
`_more_tags` and `__sklearn_tags__`:

``` py
class MyEstimator(BaseEstimator):
    def _more_tags(self):
        return {"non_deterministic": True, "poor_score": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        tags.regressor_tags.poor_score = True
        return tags
```

In order to get the tags of a given estimator, you can use the `get_tags` function:

``` py
from sklearn_compat.utils import get_tags

tags = get_tags(MyEstimator())
```

Which uses `sklearn.utils.get_tags` under the hood from scikit-learn 1.6+.

In case you want to extend the tags, you can inherit from the available tags:

```py
from sklearn_compat.utils._tags import Tags, InputTags

class MyInputTags(InputTags):
    dataframe: bool = False

class MyEstimator(BaseEstimator):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags = MyInputTags(
            one_d_array=tags.input_tags.one_d_array,
            two_d_array=tags.input_tags.two_d_array,
            sparse=tags.input_tags.sparse,
            category=True,
            dataframe=True,
            string=tags.input_tags.string,
            dict=tags.input_tags.dict,
            positive_only=tags.input_tags.positive_only,
            allow_nan=tags.input_tags.allow_nan,
            pairwise=tags.input_tags.pairwise,
        )
        return tags
```


#### `check_estimator` and `parametrize_with_checks` functions

The new tags don't include a `_xfail_checks` tags, and instead, the tests which are
expected to fail are directly passed to the `check_estimator` and
`parametrize_with_checks` functions. The two functions available in this package are
compatible with the new signature, and patch the estimator in older scikit-learn
versions to include the expected failed checks in their tags so that you don't need
to include them both in your tests and in your `_xfail_checks` old tags.


``` py
from sklearn_compat.utils.testing import parametrize_with_checks
from mypackage.myestimator import MyEstimator1, MyEstimator2

EXPECTED_FAILED_CHECKS = {
    "MyEstimator1": {"check_name1": "reason1", "check_name2": "reason2"},
    "MyEstimator2": {"check_name3": "reason3"},
}

@parametrize_with_checks([MyEstimator1(), MyEstimator2()],
                        expected_failed_checks=lambda est: EXPECTED_FAILED_CHECKS.get(
                            est.__class__.__name__, {}
                        )
)
def test_my_estimator(estimator, check):
    check(estimator)
```

### Upgrading to scikit-learn 1.5

In scikit-learn 1.5, many developer utilities have been moved to dedicated modules.
We provide a compatibility layer such that you don't have to check the version or try
to import the utilities from different modules.

In the future, when supporting scikit-learn 1.6+, you will have to change the import
from:

``` py
from sklearn_compat.utils._indexing import _safe_indexing
```

to

``` py
from sklearn.utils._indexing import _safe_indexing
```

Thus, the module path will already be correct. Now, we will go into details for
each module and function impacted.

#### `extmath` module

The function `safe_sqr` and `_approximate_mode` have been moved from `sklearn.utils` to
`sklearn.utils.extmath`.

So some code looking like this:

``` py
from sklearn.utils import safe_sqr, _approximate_mode

safe_sqr(np.array([1, 2, 3]))
_approximate_mode(class_counts=np.array([4, 2]), n_draws=3, rng=0)
```

becomes:

``` py
from sklearn_compat.utils.extmath import safe_sqr, _approximate_mode

safe_sqr(np.array([1, 2, 3]))
_approximate_mode(class_counts=np.array([4, 2]), n_draws=3, rng=0)
```

#### `type_of_target` function

The function `type_of_target` accepts a new parameter `raise_unknown`. This parameter is
available in the `sklearn_compat.utils.multiclass.type_of_target` function.

```py
from sklearn_compat.utils.multiclass import type_of_target

y = []
# raise an error with unknown target type
type_of_target(y, raise_unknown=True)
```

#### `fixes` module

The functions `_in_unstable_openblas_configuration`, `_IS_32BIT` and `_IS_WASM` have
been moved from `sklearn.utils` to `sklearn.utils.fixes`.

So the following code:

``` py
from sklearn.utils import (
    _in_unstable_openblas_configuration,
    _IS_32BIT,
    _IS_WASM,
)

_in_unstable_openblas_configuration()
print(_IS_32BIT)
print(_IS_WASM)
```

becomes:

``` py
from sklearn_compat.utils.fixes import (
    _in_unstable_openblas_configuration,
    _IS_32BIT,
    _IS_WASM,
)

_in_unstable_openblas_configuration()
print(_IS_32BIT)
print(_IS_WASM)
```

#### `validation` module

The function `_to_object_array` has been moved from `sklearn.utils` to
`sklearn.utils.validation`.

So the following code:

``` py
from sklearn.utils import _to_object_array

_to_object_array([np.array([0]), np.array([1])])
```

becomes:

``` py
from sklearn_compat.utils.validation import _to_object_array

_to_object_array([np.array([0]), np.array([1])])
```

#### `_chunking` module

The functions `gen_batches`, `gen_even_slices` and `get_chunk_n_rows` have been moved
from `sklearn.utils` to `sklearn.utils._chunking`. The function `chunk_generator` has
been moved to `sklearn.utils._chunking` as well but was renamed from `_chunk_generator`
to `chunk_generator`.

So the following code:

``` py
from sklearn.utils import (
    _chunk_generator as chunk_generator,
    gen_batches,
    gen_even_slices,
    get_chunk_n_rows,
)

_chunk_generator(range(10), 3)
gen_batches(7, 3)
gen_even_slices(10, 1)
get_chunk_n_rows(10)
```

becomes:

``` py
from sklearn_compat.utils._chunking import (
    chunk_generator, gen_batches, gen_even_slices, get_chunk_n_rows,
)

chunk_generator(range(10), 3)
gen_batches(7, 3)
gen_even_slices(10, 1)
get_chunk_n_rows(10)
```

#### `_indexing` module

The utility functions `_determine_key_type`, `_safe_indexing`, `_safe_assign`,
`_get_column_indices`, `resample` and `shuffle` have been moved from `sklearn.utils` to
`sklearn.utils._indexing`.

So the following code:

``` py
import numpy as np
import pandas as pd
from sklearn.utils import (
    _get_column_indices,
    _safe_indexing,
    _safe_assign,
    resample,
    shuffle,
)

_determine_key_type(np.arange(10))

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
_get_column_indices(df, key="b")
_safe_indexing(df, 1, axis=1)
_safe_assign(df, 1, np.array([7, 8, 9]))

array = np.arange(10)
resample(array, n_samples=20, replace=True, random_state=0)
shuffle(array, random_state=0)
```

becomes:

``` py
import numpy as np
import pandas as pd
from sklearn_compat.utils._indexing import (
    _determine_key_type,
    _safe_indexing,
    _safe_assign,
    _get_column_indices,
    resample,
    shuffle,
)

_determine_key_type(np.arange(10))

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
_get_column_indices(df, key="b")
_safe_indexing(df, 1, axis=1)
_safe_assign(df, 1, np.array([7, 8, 9]))

array = np.arange(10)
resample(array, n_samples=20, replace=True, random_state=0)
shuffle(array, random_state=0)
```

#### `_mask` module

The functions `safe_mask`, `axis0_safe_slice` and `indices_to_mask` have been moved from
`sklearn.utils` to `sklearn.utils._mask`.

So the following code:

``` py
from sklearn.utils import safe_mask, axis0_safe_slice, indices_to_mask

safe_mask(data, condition)
axis0_safe_slice(X, mask, X.shape[0])
indices_to_mask(indices, 5)
```

becomes:

``` py
from sklearn_compat.utils._mask import safe_mask, axis0_safe_slice, indices_to_mask

safe_mask(data, condition)
axis0_safe_slice(X, mask, X.shape[0])
indices_to_mask(indices, 5)
```

#### `_missing` module

The functions `is_scalar_nan` have been moved from `sklearn.utils` to
`sklearn.utils._missing`. The function `_is_pandas_na` has been moved to
`sklearn.utils._missing` as well and renamed to `is_pandas_na`.

So the following code:

``` py
from sklearn.utils import is_scalar_nan, _is_pandas_na

is_scalar_nan(float("nan"))
_is_pandas_na(float("nan"))
```

becomes:

``` py
from sklearn_compat.utils._missing import is_scalar_nan, is_pandas_na

is_scalar_nan(float("nan"))
is_pandas_na(float("nan"))
```

#### `_user_interface` module

The function `_print_elapsed_time` has been moved from `sklearn.utils` to
`sklearn.utils._user_interface`.

So the following code:

``` py
from sklearn.utils import _print_elapsed_time

with _print_elapsed_time("sklearn_compat", "testing"):
    time.sleep(0.1)
```

becomes:

``` py
from sklearn_compat.utils._user_interface import _print_elapsed_time

with _print_elapsed_time("sklearn_compat", "testing"):
    time.sleep(0.1)
```

#### `_optional_dependencies` module

The functions `check_matplotlib_support` and `check_pandas_support` have been moved from
`sklearn.utils` to `sklearn.utils._optional_dependencies`.

So the following code:

``` py
from sklearn.utils import check_matplotlib_support, check_pandas_support

check_matplotlib_support("sklearn_compat")
check_pandas_support("sklearn_compat")
```

becomes:

``` py
from sklearn_compat.utils._optional_dependencies import (
    check_matplotlib_support, check_pandas_support
)

check_matplotlib_support("sklearn_compat")
check_pandas_support("sklearn_compat")
```

### Upgrading to scikit-learn 1.4

#### `process_routing` and `_raise_for_params` functions

The signature of the `process_routing` function changed in scikit-learn 1.4. You can
import the function from `sklearn_compat.utils.metadata_routing`. The pattern will
change from:

``` py
from sklearn.utils.metadata_routing import process_routing

class MetaEstimator(BaseEstimator):
    def fit(self, X, y, sample_weight=None, **fit_params):
        params = process_routing(self, "fit", fit_params, sample_weight=sample_weight)
        return self
```

becomes:

``` py
from sklearn_compat.utils.metadata_routing import process_routing

class MetaEstimator(BaseEstimator):
    def fit(self, X, y, sample_weight=None, **fit_params):
        params = process_routing(self, "fit", sample_weight=sample_weight, **fit_params)
        return self
```

The `_raise_for_params` function was also introduced in scikit-learn 1.4. You can import
it from `sklearn_compat.utils.metadata_routing`.

``` py
from sklearn_compat.utils.metadata_routing import _raise_for_params

_raise_for_params(params, self, "fit")
```

#### Upgrading to scikit-learn 1.2

### Parameter validation

scikit-learn introduced a new way to validate parameters at `fit` time. The recommended
way to support this feature in scikit-learn 1.2+ is to inherit from
`sklearn.base.BaseEstimator` and decorate the `fit` method using the decorator
`sklearn.base._fit_context`. For functions, the decorator to use is
`sklearn.utils._param_validation.validate_params`.

We provide the function `sklearn_compat.base._fit_context` such that you can always
decorate the `fit` method of your estimator. Equivalently, you can use the function
`sklearn_compat.utils._param_validation.validate_params` to validate the parameters
of your function.

## Contributing

You can contribute to this package by:

- reporting an incompatibility with a scikit-learn version on the
  [issue tracker](https://github.com/sklearn-compat/sklearn-compat/issues). We will
  do our best to provide a compatibility layer.
- opening a [pull-request](https://github.com/sklearn-compat/sklearn-compat/pulls) to
  add a compatibility layer that you encountered when writing your scikit-learn
  compatible estimator.

Be aware that to be able to provide `sklearn-compat` as a vendorable package and a
dependency, all the changes are implemented in the
`src/sklearn_compat/_sklearn_compat.py` (indeed not the nicest experience). Then, we
need to import the changes made in this file in the submodules to use `sklearn-compat`
as a dependency.
