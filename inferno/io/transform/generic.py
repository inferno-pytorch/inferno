from .base import Transform


class Normalize(Transform):
    def __init__(self, eps=1e-4, **super_kwargs):
        """Normalizes input to zero mean unit variance."""
        super(Normalize, self).__init__(**super_kwargs)
        self.eps = eps

    def tensor_function(self, tensor):
        tensor = (tensor - tensor.mean())/(tensor.std() + self.eps)
        return tensor


class NormalizeRange(Transform):
    """Normalizes input by a constant."""
    def __init__(self, normalize_by=255., **super_kwargs):
        super(NormalizeRange, self).__init__(**super_kwargs)
        self.normalize_by = normalize_by

    def tensor_function(self, tensor):
        return tensor / self.normalize_by
