import numpy as np
import torch
from torch.autograd import Variable


class Initializer(object):
    """
    Base class for all initializers.
    """

    VALID_LAYERS = {'Conv1d', 'Conv2d', 'Conv3d',
                    'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d',
                    'Linear', 'Bilinear',
                    'Embedding'}

    def __call__(self, module):
        module_class_name = module.__class__.__name__
        if module_class_name in self.VALID_LAYERS:
            # Apply to weight and bias
            try:
                if hasattr(module, 'weight'):
                    self.call_on_weight(module.weight.data)
            except NotImplementedError:
                # Don't cry if it's not implemented
                pass

            try:
                if hasattr(module, 'bias'):
                    self.call_on_bias(module.bias.data)
            except NotImplementedError:
                pass

        return module

    def call_on_bias(self, tensor):
        return self.call_on_tensor(tensor)

    def call_on_weight(self, tensor):
        return self.call_on_tensor(tensor)

    def call_on_tensor(self, tensor):
        raise NotImplementedError

    @classmethod
    def initializes_weight(cls):
        return 'call_on_tensor' in cls.__dict__ or 'call_on_weight' in cls.__dict__

    @classmethod
    def initializes_bias(cls):
        return 'call_on_tensor' in cls.__dict__ or 'call_on_bias' in cls.__dict__


class Initialization(Initializer):
    def __init__(self, weight_initializer=None, bias_initializer=None):
        if weight_initializer is None:
            self.weight_initializer = Initializer()
        else:
            assert isinstance(weight_initializer, Initializer) \
                   and weight_initializer.initializes_weight
            self.weight_initializer = weight_initializer
        if bias_initializer is None:
            self.bias_initializer = Initializer()
        else:
            assert isinstance(bias_initializer, Initializer) \
                   and bias_initializer.initializes_bias
            self.bias_initializer = bias_initializer

    def __call__(self, module):
        module_class_name = module.__class__.__name__
        if 'Conv' in module_class_name or 'Linear' in module_class_name:
            # Apply to weight and bias
            try:
                self.weight_initializer.call_on_weight(module.weight.data)
            except NotImplementedError:
                # Don't cry if it's not implemented
                pass

            try:
                self.bias_initializer.call_on_bias(module.bias.data)
            except NotImplementedError:
                pass

        return module

    def call_on_weight(self, tensor):
        return self.weight_initializer.call_on_weight(tensor)

    def call_on_bias(self, tensor):
        return self.bias_initializer.call_on_bias(tensor)


class WeightInitFunction(Initializer):
    def __init__(self, init_function, *init_function_args, **init_function_kwargs):
        super(WeightInitFunction, self).__init__()
        assert callable(init_function)
        self.init_function = init_function
        self.init_function_args = init_function_args
        self.init_function_kwargs = init_function_kwargs

    def call_on_weight(self, tensor):
        return self.init_function(tensor, *self.init_function_args, **self.init_function_kwargs)


class BiasInitFunction(Initializer):
    def __init__(self, init_function, *init_function_args, **init_function_kwargs):
        super(BiasInitFunction, self).__init__()
        assert callable(init_function)
        self.init_function = init_function
        self.init_function_args = init_function_args
        self.init_function_kwargs = init_function_kwargs

    def call_on_bias(self, tensor):
        return self.init_function(tensor, *self.init_function_args, **self.init_function_kwargs)


class TensorInitFunction(Initializer):
    def __init__(self, init_function, *init_function_args, **init_function_kwargs):
        super(TensorInitFunction, self).__init__()
        assert callable(init_function)
        self.init_function = init_function
        self.init_function_args = init_function_args
        self.init_function_kwargs = init_function_kwargs

    def call_on_tensor(self, tensor):
        return self.init_function(tensor, *self.init_function_args, **self.init_function_kwargs)

