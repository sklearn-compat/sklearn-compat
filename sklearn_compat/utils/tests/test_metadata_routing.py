import numpy as np
from sklearn import config_context
from sklearn.base import BaseEstimator, clone
from sklearn.datasets import make_classification
from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping

from sklearn_compat.utils.metadata_routing import process_routing


def test_process_routing():
    """Check the backward compatibility of the `process_routing` function."""

    class Estimator(BaseEstimator):
        def fit(self, X, y, sample_weight=None):
            self.sample_weight_ = sample_weight
            return self

    class MetaEstimator(BaseEstimator):

        def __init__(self, estimator):
            self.estimator = estimator

        def fit(self, X, y, **fit_params):
            params = process_routing(self, "fit", **fit_params)
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, y, **params.estimator.fit)
            return self

        def get_metadata_routing(self):
            router = MetadataRouter(owner=self.__class__.__name__)
            router.add(
                estimator=self.estimator,
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),
            )
            return router

    with config_context(enable_metadata_routing=True):
        X, y = make_classification(n_samples=10, n_features=5, random_state=0)
        sample_weight = np.ones(len(y))
        estimator = Estimator()
        estimator.set_fit_request(sample_weight=True)
        meta_estimator = MetaEstimator(estimator)
        meta_estimator.fit(X, y, sample_weight=sample_weight)
        np.testing.assert_array_equal(
            meta_estimator.estimator_.sample_weight_, sample_weight
        )
