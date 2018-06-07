from .base import Callback

class TQDMProgressBar(Callback):
    def __init__(self, *args, **kwargs):
        super(TQDMProgressBar, self).__init__(*args, **kwargs)

    def bind_trainer(self, *args, **kwargs):
        super(TQDMProgressBar, self).bind_trainer(*args, **kwargs)
        self.trainer.console.warning("tqdm is not installed. will fall back to normal stdout console.")

    def begin_of_fit(self, **_):
        pass
