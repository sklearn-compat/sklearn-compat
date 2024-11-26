from ._version import parse_version, sklearn_version

if sklearn_version < parse_version("1.4"):
    def process_routing(_obj, _method, /, **kwargs):
        from sklearn.utils.metadata_routing import process_routing as _process_routing

        return _process_routing(_obj, _method, kwargs)

else:
    from sklearn.utils.metadata_routing import process_routing  # noqa: F401
