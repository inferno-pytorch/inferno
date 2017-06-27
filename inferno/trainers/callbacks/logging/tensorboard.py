try:
    import tensorflow as tf
except ImportError:
    tf = None
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import numpy as np

from .base import Logger
from ....utils import torch_utils as tu


class TensorboardLogger(Logger):
    # Borrowed from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    def __init__(self, log_directory=None, **config):
        assert tf is not None
        assert plt is not None
        super(TensorboardLogger, self).__init__(log_directory=log_directory)
        self._writer = None
        self._config = config

    @property
    def writer(self):
        if self._writer is None:
            self._writer = tf.summary.FileWriter(self.log_directory)
        return self._writer

    def end_of_training_iteration(self, **_):
        # Fetch from trainer
        training_loss = self.trainer.get_state('training_loss')
        training_error = self.trainer.get_state('training_error')
        training_prediction = self.trainer.get_state('training_prediction')
        training_inputs = self.trainer.get_state('training_inputs')
        training_target = self.trainer.get_state('training_target')
        # Extract floats from torch tensors if necessary
        if tu.is_tensor(training_loss):
            training_loss = training_loss.float()[0]
        if tu.is_tensor(training_error):
            training_error = training_error.float()[0]
        self.log_scalar('training_loss', training_loss, self.trainer.iteration_count)
        self.log_scalar('training_error', training_error, self.trainer.iteration_count)

    def extract_images_from_batch(self, batch):
        if tu.is_image_tensor(batch):
            # Convert to numpy
            batch = batch.numpy()
            # TODO Continue
        pass

    def log_scalar(self, tag, value, step):
        """
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_images(self, tag, images, step):
        """Logs a list of images."""

        image_summaries = []
        for image_num, image in enumerate(images):
            # Write the image to a string
            s = StringIO()
            plt.imsave(s, image, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=image.shape[0],
                                       width=image.shape[1])
            # Create a Summary value
            image_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, image_num),
                                                    image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=image_summaries)
        self.writer.add_summary(summary, step)

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()