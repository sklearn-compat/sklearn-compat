import time

from sklearn_compat.utils._user_interface import _print_elapsed_time


def test_print_elapsed_time():
    """Check that we can import `_print_elapsed_time` from the right module.

    This change has been done in scikit-learn 1.5.
    """
    import sys
    from io import StringIO

    stdout = StringIO()
    sys.stdout = stdout
    with _print_elapsed_time("sklearn_compat", "testing"):
        time.sleep(0.1)
    sys.stdout = sys.__stdout__
    output = stdout.getvalue()
    assert output.startswith("[sklearn_compat] .....")
