# Minimal kaolin shim — only check_tensor is used in the inference path.
import sys
import types


def _make(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


def _install():
    try:
        import kaolin  # noqa: F401 — if real kaolin is installed, skip
        return
    except ImportError:
        pass
    k = _make("kaolin")
    ut = _make("kaolin.utils")
    test = _make("kaolin.utils.testing")

    def check_tensor(*a, **kw):
        return True

    test.check_tensor = check_tensor
    k.utils = ut
    ut.testing = test


_install()
