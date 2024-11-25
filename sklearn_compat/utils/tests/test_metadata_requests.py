import numpy as np
from sklearn import config_context
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from sklearn_compat.base import MetadataRequesterMixin
from sklearn_compat.utils.metadata_routing import (
    MethodMapping,
    MetadataRouter,
    process_routing,
)


class MyMetaEstimator(MetadataRequesterMixin, MetaEstimatorMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        routed_params = process_routing(self, "fit", **fit_params)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **routed_params.estimator.fit)
        return self

    def _get_estimator(self):
        return self.estimator

    def get_metadata_routing(self):
        router = MetadataRouter(owner=self.__class__.__name__)
        router.add(
            estimator=self._get_estimator(),
            method_mapping=MethodMapping().add(caller="fit", callee="fit"),
        )
        return router


def test_metadata_routing_meta_estimator():
    X, y = make_classification(n_samples=100, n_features=20, random_state=0)
    rng = np.random.default_rng(0)
    sample_weight = rng.uniform(size=y.shape)

    with config_context(enable_metadata_routing=True):
        estimator = LogisticRegression().set_fit_request(sample_weight=True)
        meta_estimator = MyMetaEstimator(estimator)
        meta_estimator.fit(X, y, sample_weight=sample_weight)
