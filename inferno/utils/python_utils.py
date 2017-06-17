"""Utility functions with no external dependencies."""


def to_iterable(x):
    if not isinstance(x, (list, tuple)):
        return list(x)
    else:
        return x


def from_iterable(x):
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return x[0]
    else:
        return x