from .. utils.singleton import Singleton
from tensorboardX import SummaryWriter
from functools import partialmethod

@Singleton
class TensorboardLogger(object):

    def __init__(self):
        print("construct")
        self.summary_writer = None



    def setup(self, *args, **kwargs ):
        if self.summary_writer is not None:
            raise RuntimeError("set_up can only be called once")
        self.summary_writer = SummaryWriter(*args, **kwargs)

    def add_audio(self, *args, **kwargs ):
        return self.summary_writer.add_audio(*args, **kwargs)
    def add_custom_scalars(self, *args, **kwargs ):
        return self.summary_writer.add_custom_scalars(*args, **kwargs)
    def add_custom_scalars_marginchart(self, *args, **kwargs ):
        return self.summary_writer.add_custom_scalars_marginchart(*args, **kwargs)        
    def add_custom_scalars_multilinechart(self, *args, **kwargs ):
        return self.summary_writer.add_custom_scalars_multilinechart(*args, **kwargs)
    def add_figure(self, *args, **kwargs ):
        return self.summary_writer.add_figure(*args, **kwargs)
    def add_graph(self, *args, **kwargs ):
        return self.summary_writer.add_graph(*args, **kwargs)
    def add_histogram(self, *args, **kwargs ):
        return self.summary_writer.add_histogram(*args, **kwargs)
    def add_histogram_raw(self, *args, **kwargs ):
        return self.summary_writer.add_histogram_raw(*args, **kwargs)
    def add_hparams(self, *args, **kwargs ):
        return self.summary_writer.add_hparams(*args, **kwargs)
    def add_image(self, *args, **kwargs ):
        return self.summary_writer.add_image(*args, **kwargs)
    def add_mesh(self, *args, **kwargs ):
        return self.summary_writer.add_mesh(*args, **kwargs)
    def add_pr_curve(self, *args, **kwargs ):
        return self.summary_writer.add_pr_curve(*args, **kwargs)        
    def add_pr_curve_raw(self, *args, **kwargs ):
        return self.summary_writer.add_pr_curve_raw(*args, **kwargs)
    def add_scalar(self, *args, **kwargs ):
        return self.summary_writer.add_scalar(*args, **kwargs)
    def add_scalars(self, *args, **kwargs ):
        return self.summary_writer.add_scalars(*args, **kwargs)
    def add_text(self, *args, **kwargs ):
        return self.summary_writer.add_text(*args, **kwargs)
    def add_video(self, *args, **kwargs ):
        return self.summary_writer.add_video(*args, **kwargs)
    def add_embedding(self, *args, **kwargs ):
        return self.summary_writer.add_embedding(*args, **kwargs)
