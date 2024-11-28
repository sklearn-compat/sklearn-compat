from sklearn_compat.utils._test_common.instance_generator import _construct_instances
from sklearn.linear_model import LinearRegression

def test__construct_instances():
    list(iter(_construct_instances(LinearRegression)))
