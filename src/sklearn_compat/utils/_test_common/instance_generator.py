try:
    from sklearn.utils._test_common.instance_generator import _construct_instances
except ImportError:
    from sklearn.utils.estimator_checks import _construct_instance

    def _construct_instances(Estimator):
        yield _construct_instance(Estimator)
