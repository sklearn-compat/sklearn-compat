from ._version import parse_version, sklearn_version


if sklearn_version < parse_version("1.3"):

    def _is_fitted(estimator, attributes=None, all_or_any=all):
        """Determine if an estimator is fitted

        Parameters
        ----------
        estimator : estimator instance
            Estimator instance for which the check is performed.

        attributes : str, list or tuple of str, default=None
            Attribute name(s) given as string or a list/tuple of strings
            Eg.: ``["coef_", "estimator_", ...], "coef_"``

            If `None`, `estimator` is considered fitted if there exist an
            attribute that ends with a underscore and does not start with double
            underscore.

        all_or_any : callable, {all, any}, default=all
            Specify whether all or any of the given attributes must exist.

        Returns
        -------
        fitted : bool
            Whether the estimator is fitted.
        """
        if attributes is not None:
            if not isinstance(attributes, (list, tuple)):
                attributes = [attributes]
            return all_or_any([hasattr(estimator, attr) for attr in attributes])

        if hasattr(estimator, "__sklearn_is_fitted__"):
            return estimator.__sklearn_is_fitted__()

        fitted_attrs = [
            v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
        ]
        return len(fitted_attrs) > 0

else:
    from sklearn.utils.validation import _is_fitted


if sklearn_version < parse_version("1.6"):
    def validate_data(_estimator, /, **kwargs):
        if "ensure_all_finite" in kwargs:
            force_all_finite = kwargs.pop("ensure_all_finite")
        else:
            force_all_finite = True
        return _estimator._validate_data(**kwargs, force_all_finite=force_all_finite)
else:
    from sklearn.utils.validation import validate_data


if sklearn_version < parse_version("1.6"):
    def _check_n_features(estimator, X, *, reset):
        return estimator._check_n_features(X, reset=reset)
else:
    from sklearn.utils.validation import _check_n_features

if sklearn_version < parse_version("1.6"):
    def _check_feature_names(estimator, X, *, reset):
        return estimator._check_feature_names(X, reset=reset)
else:
    from sklearn.utils.validation import _check_feature_names