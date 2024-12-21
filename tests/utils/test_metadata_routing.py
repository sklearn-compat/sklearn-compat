import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone

from sklearn_compat._sklearn_compat import parse_version, sklearn_version
from sklearn_compat.utils.metadata_routing import _raise_for_params, process_routing


@pytest.mark.skipif(
    sklearn_version < parse_version("1.3"),
    reason="Metadata routing was introduced in scikit-learn 1.3",
)
def test_process_routing():
    from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping

    class ExampleClassifier(ClassifierMixin, BaseEstimator):
        def fit(self, X, y, sample_weight=None):
            self.sample_weight_ = sample_weight
            # all classifiers need to expose a classes_ attribute once they're fit.
            self.classes_ = np.array([0, 1])
            return self

    class MetaClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
        def __init__(self, estimator):
            self.estimator = estimator

        def fit(self, X, y, sample_weight=None):
            routed_params = process_routing(self, "fit", sample_weight=sample_weight)
            self.estimator_ = clone(self.estimator).fit(
                X, y, **routed_params.estimator.fit
            )
            return self

        def get_metadata_routing(self):
            router = MetadataRouter(owner=self.__class__.__name__).add(
                estimator=self.estimator,
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),
            )
            return router

    with config_context(enable_metadata_routing=True):
        est = MetaClassifier(ExampleClassifier().set_fit_request(sample_weight=True))
        sample_weight = [1, 2, 3]
        est.fit(None, None, sample_weight=sample_weight)
        assert est.estimator_.sample_weight_ == sample_weight


@pytest.mark.skipif(
    sklearn_version < parse_version("1.3"),
    reason="Metadata routing was introduced in scikit-learn 1.3",
)
def test_raise_for_params():
    class DummyEstimator(BaseEstimator):
        pass

    est = DummyEstimator()
    params = {"invalid_param": 42}

    with pytest.raises(ValueError, match="Passing extra keyword arguments"):
        _raise_for_params(params, est, "fit")


@pytest.mark.skipif(
    sklearn_version >= parse_version("1.3"),
    reason="Metadata routing is implemented",
)
def test__raise_for_params_not_implemented():
    with pytest.raises(NotImplementedError):
        _raise_for_params(None, None, "fit")
    with pytest.raises(NotImplementedError):
        process_routing(None, "fit")
