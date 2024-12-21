from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from sklearn_compat.base import is_clusterer


def test_is_clusterer():
    assert is_clusterer(KMeans())
    assert not is_clusterer(LinearRegression())
