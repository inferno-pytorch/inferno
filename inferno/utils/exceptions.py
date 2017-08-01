"""Exceptions and Error Handling"""


def assert_(condition, message='', exception_type=AssertionError):
    """Like assert, but with arbitrary exception types."""
    if not condition:
        raise exception_type(message)


class ShapeError(ValueError):
    pass


class NotTorchModuleError(TypeError):
    pass
