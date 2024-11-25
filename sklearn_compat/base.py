import functools

from sklearn._config import config_context, get_config

from .utils.validation import _is_fitted
from .utils._metadata_requests import _MetadataRequester
from .utils._param_validation import validate_parameter_constraints
from .utils._version import parse_version as parse_version, sklearn_version


class MetadataRequesterMixin(_MetadataRequester):
    """Mixin class to enable metadata routing."""
    pass


class ParamsValidationMixin:
    """Mixin class to validate parameters."""

    def _validate_params(self):
        """Validate types and values of constructor parameters.

        The expected type and values must be defined in the `_parameter_constraints`
        class attribute, which is a dictionary `param_name: list of constraints`. See
        the docstring of `validate_parameter_constraints` for a description of the
        accepted constraints.
        """
        if hasattr(self, "_parameter_constraints"):
            validate_parameter_constraints(
                self._parameter_constraints,
                self.get_params(deep=False),
                caller_name=self.__class__.__name__,
            )


if sklearn_version < parse_version("1.4"):
    def _fit_context(*, prefer_skip_nested_validation):
        """Decorator to run the fit methods of estimators within context managers.

        Parameters
        ----------
        prefer_skip_nested_validation : bool
            If True, the validation of parameters of inner estimators or functions
            called during fit will be skipped.

            This is useful to avoid validating many times the parameters passed by the
            user from the public facing API. It's also useful to avoid validating
            parameters that we pass internally to inner functions that are guaranteed to
            be valid by the test suite.

            It should be set to True for most estimators, except for those that receive
            non-validated objects as parameters, such as meta-estimators that are given
            estimator objects.

        Returns
        -------
        decorated_fit : method
            The decorated fit method.
        """

        def decorator(fit_method):
            @functools.wraps(fit_method)
            def wrapper(estimator, *args, **kwargs):
                global_skip_validation = get_config()["skip_parameter_validation"]

                # we don't want to validate again for each call to partial_fit
                partial_fit_and_fitted = (
                    fit_method.__name__ == "partial_fit" and _is_fitted(estimator)
                )

                if not global_skip_validation and not partial_fit_and_fitted:
                    estimator._validate_params()

                with config_context(
                    skip_parameter_validation=(
                        prefer_skip_nested_validation or global_skip_validation
                    )
                ):
                    return fit_method(estimator, *args, **kwargs)

            return wrapper

        return decorator
else:
    from sklearn.base import _fit_context  # noqa: F401
