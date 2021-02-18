from .. utils.singleton import Singleton
from .. utils import train_utils as tu
from tensorboardX import SummaryWriter


@Singleton
class TensorboardSummaryWriter(object):

    def __init__(self):
        self.summary_writer = None
        self.trainer = None

        self._add_audio = None
        self._add_custom_scalars = None
        self._add_custom_scalars_marginchart = None
        self._add_custom_scalars_multilinechart = None
        self._add_figure = None
        self._add_graph = None
        self._add_histogram = None
        self._add_histogram_raw = None
        self._add_hparams = None
        self._add_image = None
        self._add_mesh = None
        self._add_pr_curve = None
        self._add_pr_curve_raw = None
        self._add_scalar = None
        self._add_scalars = None
        self._add_text = None
        self._add_video = None
        self._add_embedding = None

    def setup(self, trainer, *args, **kwargs ):
        if self.summary_writer is not None:
            raise RuntimeError("set_up can only be called once")
        self.summary_writer = SummaryWriter(*args, **kwargs)
        self.trainer = trainer

    def _match(self,frequency):
        if frequency is None:
            return True
        else:
            return frequency.match(
                epoch_count=self.trainer.epoch_count,
                iteration_count=self.trainer.iteration_count,
                persistent=True
            )

    def add_audio(self, *args, **kwargs ):
        if self._match(self._add_audio):
            return self.summary_writer.add_audio(*args, **kwargs, global_step=self.trainer.iteration_count)
    def add_custom_scalars(self, *args, **kwargs ):
        if self._match(self._add_custom_scalars):
            return self.summary_writer.add_custom_scalars(*args, **kwargs, global_step=self.trainer.iteration_count)
    def add_custom_scalars_marginchart(self, *args, **kwargs ):
        if self._match(self._add_custom_scalars_marginchart):
            return self.summary_writer.add_custom_scalars_marginchart(*args, **kwargs, global_step=self.trainer.iteration_count)        
    def add_custom_scalars_multilinechart(self, *args, **kwargs ):
        if self._match(self._add_custom_scalars_multilinechart):
            return self.summary_writer.add_custom_scalars_multilinechart(*args, **kwargs, global_step=self.trainer.iteration_count)
    def add_figure(self, *args, **kwargs ):
        if self._match(self._add_figure):
            return self.summary_writer.add_figure(*args, **kwargs, global_step=self.trainer.iteration_count)
    def add_graph(self, *args, **kwargs ):
        if self._match(self._add_graph):
            return self.summary_writer.add_graph(*args, **kwargs, global_step=self.trainer.iteration_count)
    def add_histogram(self, *args, **kwargs ):
        if self._match(self._add_histogram):
            return self.summary_writer.add_histogram(*args, **kwargs, global_step=self.trainer.iteration_count)
    def add_histogram_raw(self, *args, **kwargs ):
        if self._match(self._add_histogram_raw):
            return self.summary_writer.add_histogram_raw(*args, **kwargs, global_step=self.trainer.iteration_count)
    def add_hparams(self, *args, **kwargs ):
        if self._match(self._add_hparams):
            return self.summary_writer.add_hparams(*args, **kwargs, global_step=self.trainer.iteration_count)
    def add_image(self, *args, **kwargs ):
        if self._match(self._add_image):
            return self.summary_writer.add_image(*args, **kwargs, global_step=self.trainer.iteration_count)
    def add_mesh(self, *args, **kwargs ):
        if self._match(self._add_mesh):
            return self.summary_writer.add_mesh(*args, **kwargs, global_step=self.trainer.iteration_count)
    def add_pr_curve(self, *args, **kwargs ):
        if self._match(self._add_pr_curve):
            return self.summary_writer.add_pr_curve(*args, **kwargs, global_step=self.trainer.iteration_count)        
    def add_pr_curve_raw(self, *args, **kwargs ):
        if self._match(self._add_pr_curve_raw):
            return self.summary_writer.add_pr_curve_raw(*args, **kwargs, global_step=self.trainer.iteration_count)
    def add_scalar(self, *args, **kwargs ):
        if self._match(self._add_scalar):
            return self.summary_writer.add_scalar(*args, **kwargs, global_step=self.trainer.iteration_count)
    def add_scalars(self, *args, **kwargs ):
        if self._match(self._add_scalars):
            return self.summary_writer.add_scalars(*args, **kwargs, global_step=self.trainer.iteration_count)
    def add_text(self, *args, **kwargs ):
        if self._match(self._add_text):
            return self.summary_writer.add_text(*args, **kwargs, global_step=self.trainer.iteration_count)
    def add_video(self, *args, **kwargs ):
        if self._match(self._add_video):
            return self.summary_writer.add_video(*args, **kwargs, global_step=self.trainer.iteration_count)
    def add_embedding(self, *args, **kwargs ):
        if self._match(self._add_embedding):
            return self.summary_writer.add_embedding(*args, **kwargs, global_step=self.trainer.iteration_count)


    def add_audio_every(self, frequency):
        self._add_audio = tu.Frequency.build_from(frequency, priority='iterations')
        return self
    def add_custom_scalars_every(self, frequency):
        self._add_custom_scalars = tu.Frequency.build_from(frequency, priority='iterations')
        return self
    def add_custom_scalars_marginchart_every(self, frequency):
        self._add_custom_scalars_marginchart = tu.Frequency.build_from(frequency, priority='iterations')
        return self
    def add_custom_scalars_multilinechart_every(self, frequency):
        self._add_custom_scalars_multilinechart = tu.Frequency.build_from(frequency, priority='iterations')
        return self
    def add_figure_every(self, frequency):
        self._add_figure = tu.Frequency.build_from(frequency, priority='iterations')
        return self
    def add_graph_every(self, frequency):
        self._add_graph = tu.Frequency.build_from(frequency, priority='iterations')
        return self
    def add_histogram_every(self, frequency):
        self._add_histogram = tu.Frequency.build_from(frequency, priority='iterations')
        return self
    def add_histogram_raw_every(self, frequency):
        self._add_histogram_raw = tu.Frequency.build_from(frequency, priority='iterations')
        return self
    def add_hparams_every(self, frequency):
        self._add_hparams = tu.Frequency.build_from(frequency, priority='iterations')
        return self
    def add_image_every(self, frequency):
        self._add_image = tu.Frequency.build_from(frequency, priority='iterations')
        return self
    def add_mesh_every(self, frequency):
        self._add_mesh = tu.Frequency.build_from(frequency, priority='iterations')
        return self
    def add_pr_curve_every(self, frequency):
        self._add_pr_curve = tu.Frequency.build_from(frequency, priority='iterations')
        return self
    def add_pr_curve_raw_every(self, frequency):
        self._add_pr_curve_raw = tu.Frequency.build_from(frequency, priority='iterations')
        return self
    def add_scalar_every(self, frequency):
        self._add_scalar = tu.Frequency.build_from(frequency, priority='iterations')
        return self
    def add_scalars_every(self, frequency):
        self._add_scalars = tu.Frequency.build_from(frequency, priority='iterations')
        return self
    def add_text_every(self, frequency):
        self._add_text = tu.Frequency.build_from(frequency, priority='iterations')
        return self
    def add_video_every(self, frequency):
        self._add_video = tu.Frequency.build_from(frequency, priority='iterations')
        return self
    def add_embedding_every(self, frequency):
        self._add_embedding = tu.Frequency.build_from(frequency, priority='iterations')
        return self
























