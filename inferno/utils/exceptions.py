"""Exceptions and Error Handling"""


def assert_(condition, message='', exception_type=AssertionError):
    """Like assert, but with arbitrary exception types."""
    if not condition:
        raise exception_type(message)


# ------ VALUE ERRORS ------


class ShapeError(ValueError):
    pass


class FrequencyValueError(ValueError):
    pass


# ------ TYPE ERRORS ------


class NotTorchModuleError(TypeError):
    pass


class FrequencyTypeError(TypeError):
    pass