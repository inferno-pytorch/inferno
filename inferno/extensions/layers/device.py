import torch.nn as nn
from ...utils.python_utils import from_iterable
from ...utils.exceptions import assert_, DeviceError


class DeviceTransfer(nn.Module):
    """Layer to transfer variables to a specified device."""
    def __init__(self, target_device, device_ordinal=None, async=False):
        """
        Parameters
        ----------
        target_device : {'cpu', 'cuda'}
            Device to transfer to.
        device_ordinal : int
            Device ordinal if target_device == 'cuda'.
        async : bool
            Whether to use async transfers.
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
        self.async = async

    def forward(self, *inputs):
        if self.target_device == 'cuda':
            transferred = tuple(input_.cuda(device_id=self.device_ordinal, async=self.async)
                                for input_ in inputs)
        elif self.target_device == 'cpu':
            transferred = tuple(input_.cpu() for input_ in inputs)
        else:
            raise NotImplementedError
        return from_iterable(transferred)