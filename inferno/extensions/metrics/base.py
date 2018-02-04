

class Metric(object):
    def __init__(self, is_multiscale=False):
        self.is_multiscale = is_multiscale

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, prediction, target, **kwargs):
        # For multiscale training, we only care about the 0th scale level
        # for evaluation and extract it
        if self.is_multiscale:
            prediction = prediction[0]
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
