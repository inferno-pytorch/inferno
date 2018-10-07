import torch
from torch.autograd import Variable
from .exceptions import assert_, NotTorchModuleError, ShapeError


def is_model_cuda(model):
    try:
        return next(model.parameters()).is_cuda
    except StopIteration:
        # Assuming that if a network has no parameters, it doesn't use CUDA
        return False


class ModelTester(object):
    def __init__(self, input_shape, expected_output_shape):
        self._is_cuda = False
        self.input_shape = input_shape
        self.expected_output_shape = expected_output_shape

    def cuda(self):
        self._is_cuda = True
        return self

    def get_input(self):
        with torch.no_grad():
            if self._is_cuda:
                return torch.rand(*self.input_shape, requires_grad=False).cuda(),
            else:
                return torch.rand(*self.input_shape, requires_grad=False)

    def __call__(self, model):
        # Make sure model is a model
        assert_(isinstance(model, torch.nn.Module),
                "Model is not a torch module.",
                NotTorchModuleError)
        # Transfer to cuda if required
        if not is_model_cuda(model) and self._is_cuda:
            model.cuda()
        input_ = self.get_input()
        output = model(input_)
        assert_(list(output.size()) == list(self.expected_output_shape),
                "Expected output shape {} for input shape {}, "
                "got output of shape {} instead.".format(list(self.expected_output_shape),
                                                         list(self.input_shape),
                                                         list(output.size())),
                ShapeError)
        return model


class MultiscaleModelTester(ModelTester):
    def __call__(self, model):
        # Make sure model is a model
        assert_(isinstance(model, torch.nn.Module),
                "Model is not a torch module.",
                NotTorchModuleError)
        # Transfer to cuda if required
        if not is_model_cuda(model) and self._is_cuda:
            model.cuda()
        input_ = self.get_input()
        output = model(input_)
        assert_(isinstance(output, tuple), "Expect tuple output")
        for scale in range(len(output)):
            assert_(list(output[scale].size()) == list(self.expected_output_shape[scale]),
                    "Expected output shape {} for input shape {}, "
                    "got output of shape {} instead.".format(list(self.expected_output_shape[scale]),
                                                             list(self.input_shape),
                                                             list(output[scale].size())),
                    ShapeError)
        return model
