from sklearn.utils._param_validation import validate_parameter_constraints

from .utils._version import parse_version as parse_version


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
