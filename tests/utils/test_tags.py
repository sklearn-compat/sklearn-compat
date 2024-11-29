from sklearn.base import BaseEstimator

from sklearn_compat.utils._tags import (
    get_tags,
)


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
