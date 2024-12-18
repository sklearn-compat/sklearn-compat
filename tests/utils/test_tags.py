import pytest
from sklearn.base import BaseEstimator

from sklearn_compat._sklearn_compat import parse_version, sklearn_version
from sklearn_compat.utils._tags import get_tags


def test_get_tags():
    class MyEstimator(BaseEstimator):
        def _more_tags(self):
            return {"requires_fit": False}

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.requires_fit = False
            return tags

    tags = get_tags(MyEstimator())
    assert not tags.requires_fit


@pytest.mark.skipif(
    sklearn_version >= parse_version("1.6"),
    reason="Test only relevant for sklearn < 1.6",
)
def test_patched_more_tags():
    """Non-regression test for:
    https://github.com/sklearn-compat/sklearn-compat/issues/27

    The regression can be spotted when an estimator dynamically updates the tags
    based on the inner estimator.
    """

    class MyEstimator(BaseEstimator):
        def __init__(self, allow_nan=False):
            self.allow_nan = allow_nan

        def _more_tags(self):
            return {"allow_nan": self.allow_nan}

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.input_tags.allow_nan = self.allow_nan
            return tags

    class MetaEstimator(BaseEstimator):
        def __init__(self, estimator):
            self.estimator = estimator

        def fit(self, X, y=None):
            return self

        def _more_tags(self):
            return {
                "allow_nan": get_tags(self.estimator).input_tags.allow_nan,
            }

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.input_tags.allow_nan = get_tags(self.estimator).input_tags.allow_nan
            return tags

    est_no_support_nan = MetaEstimator(estimator=MyEstimator(allow_nan=False))
    est_support_nan = MetaEstimator(estimator=MyEstimator(allow_nan=True))

    assert not get_tags(est_no_support_nan).input_tags.allow_nan
    assert get_tags(est_support_nan).input_tags.allow_nan

    from sklearn_compat._sklearn_compat import _patched_more_tags

    test_to_fail = [{"some_check": True}]
    _patched_more_tags(est_no_support_nan, expected_failed_checks=test_to_fail)
    _patched_more_tags(est_support_nan, expected_failed_checks=test_to_fail)

    # check that patching the instance to add the test to fail should not overwrite
    # other tags
    assert not get_tags(est_no_support_nan).input_tags.allow_nan
    assert get_tags(est_support_nan).input_tags.allow_nan
    # check that accessing the _xfail_checks via the old _get_tags API report the
    # test to be skipped
    assert est_no_support_nan._get_tags()["_xfail_checks"] == test_to_fail
    assert est_support_nan._get_tags()["_xfail_checks"] == test_to_fail
