

class Metric(object):

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)
