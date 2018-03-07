

class Metric(object):
    def __init__(self):
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, prediction, target, **kwargs):
        # For multi-output training, we only care about the 0th scale level
        # for evaluation and extract it
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        if isinstance(target, tuple):
            target = target[0]

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
