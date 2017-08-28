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


class DeviceError(ValueError):
    pass


class NotSetError(ValueError):
    pass


# ------ TYPE ERRORS ------


class NotTorchModuleError(TypeError):
    pass


class FrequencyTypeError(TypeError):
    pass


class DTypeError(TypeError):
    pass


# ------ LOOKUP ERRORS ------


class ClassNotFoundError(LookupError):
    pass


# ------ NOT-IMPLEMENTED ERRORS ------


class NotUnwrappableError(NotImplementedError):
    pass