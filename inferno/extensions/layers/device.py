import torch.nn as nn
from ...utils.python_utils import from_iterable, to_iterable
from ...utils.exceptions import assert_, DeviceError

__all__ = ['DeviceTransfer', 'OnDevice']
_all = __all__

class DeviceTransfer(nn.Module):
    """Layer to transfer variables to a specified device."""
    def __init__(self, target_device, device_ordinal=None, asynchron=False):
        """
        Parameters
        ----------
        target_device : {'cpu', 'cuda'}
            Device to transfer to.
        device_ordinal : int
            Device ordinal if target_device == 'cuda'.
        asynchron : bool
            Whether to use asynchronous transfers.
        """
        super(DeviceTransfer, self).__init__()
        # Validate arguments
        assert_(target_device in ['cpu', 'cuda'],
                "Target device must either be 'cpu' or 'cuda'.",
                DeviceError)
        if target_device == 'cpu':
            assert_(device_ordinal is None,
                    "'device_ordinal' must be None if target_device is 'cpu'.",
                    DeviceError)
        self.target_device = target_device
        self.device_ordinal = device_ordinal
        self.asynchron = asynchron

    def forward(self, *inputs):
        if self.target_device == 'cuda':
            transferred = tuple(input_.cuda(device_id=self.device_ordinal, asynchron=self.asynchron)
                                for input_ in inputs)
        elif self.target_device == 'cpu':
            transferred = tuple(input_.cpu() for input_ in inputs)
        else:
            raise NotImplementedError
        return from_iterable(transferred)


class OnDevice(nn.Module):
    """
    Moves a module to a device. The advantage of using this over `torch.nn.Module.cuda` is
    that the inputs are transferred to the same device as the module, enabling easy model
    parallelism.
    """
    def __init__(self, module, target_device, device_ordinal=None, asynchron=False):
        """
        Parameters
        ----------
        module : torch.nn.Module
            Module to transfer to device.
        target_device : {'cuda', 'cpu'}
            The device to move `module` to. Must be either 'cuda' or 'cpu'.
        device_ordinal : int
            Ordinal of the GPU device if `target_device = 'cuda'`.
        asynchron : bool
            Whether to use asynchronous transfers.
        """
        super(OnDevice, self).__init__()
        # Validate arguments
        assert_(target_device in ['cpu', 'cuda'],
                "Target device must either be 'cpu' or 'cuda'.",
                DeviceError)
        if target_device == 'cpu':
            assert_(device_ordinal is None,
                    "'device_ordinal' must be None if target_device is 'cpu'.",
                    DeviceError)
        self.target_device = target_device
        self.device_ordinal = device_ordinal
        self.asynchron = asynchron
        # This is a no-op if module is already in the right device
        self.device_transfer = DeviceTransfer(self.target_device,
                                              device_ordinal=self.device_ordinal,
                                              asynchron=self.asynchron)

        self.module = self.transfer_module(module)

    def transfer_module(self, module):
        if self.target_device == 'cuda':
            return module.cuda(device_id=self.device_ordinal)
        elif self.target_device == 'cpu':
            return module.cpu()
        else:
            raise NotImplementedError

    def forward(self, *inputs):
        # Transfer inputs (no-op if they're already on the right device)
        transferred = to_iterable(self.device_transfer(*inputs))
        output = self.module(*transferred)
        return output
