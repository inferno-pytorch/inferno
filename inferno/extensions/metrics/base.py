

class Metric(object):

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, prediction, target, **kwargs):
        # Make sure prediction and target live on the same device.
        # If they don't, move target to the right device.
        if not prediction.is_cuda:
            # Move to CPU
            target = target.cpu()
        else:
            # Find device to move to
            device_ordinal = prediction.get_device()
            target = target.cuda(device_ordinal)
        return self.forward(prediction, target, **kwargs)
