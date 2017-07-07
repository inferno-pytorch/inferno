try:
    import tensorflow as tf
except ImportError:
    tf = None
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO, BytesIO
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import numpy as np
from scipy.misc import toimage

from .base import Logger
from ....utils import torch_utils as tu
from ....utils import python_utils as pyu
from ....utils import train_utils as tru


class TensorboardLogger(Logger):
    # Borrowed from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514 and
    # https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py
    def __init__(self, log_directory=None, log_scalars_every=None, log_images_every=None,
                 send_image_at_batch_indices='all', send_image_at_channel_indices='all',
                 send_volume_at_z_indices='mid'):
        assert tf is not None
        assert plt is not None
        super(TensorboardLogger, self).__init__(log_directory=log_directory)
        self._log_scalars_every = None
        self._log_images_every = None
        self._writer = None
        self._config = {'image_batch_indices': send_image_at_batch_indices,
                        'image_channel_indices': send_image_at_channel_indices,
                        'volume_z_indices': send_volume_at_z_indices}
        if log_scalars_every is not None:
            self.log_scalars_every = log_scalars_every
        if log_images_every is not None:
            self.log_images_every = log_images_every

    @property
    def writer(self):
        if self._writer is None:
            self._writer = tf.summary.FileWriter(self.log_directory)
        return self._writer

    @property
    def log_scalars_every(self):
        if self._log_scalars_every is None:
            self._log_scalars_every = tru.Frequency(1, 'iterations')
        return self._log_scalars_every

    @log_scalars_every.setter
    def log_scalars_every(self, value):
        self._log_scalars_every = tru.Frequency.build_from(value)

    @property
    def log_scalars_now(self):
        # Using persistent=True in a property getter is probably not a very good idea...
        # We need to make sure that this getter is called only once per callback-call.
        return self.log_scalars_every.match(iteration_count=self.trainer.iteration_count,
                                            epoch_count=self.trainer.epoch_count,
                                            persistent=True)

    @property
    def log_images_every(self):
        if self._log_images_every is None:
            self._log_images_every = tru.Frequency(1, 'iterations')
        return self._log_images_every

    @log_images_every.setter
    def log_images_every(self, value):
        self._log_images_every = tru.Frequency.build_from(value)

    @property
    def log_images_now(self):
        # Using persistent=True in a property getter is probably not a very good idea...
        # We need to make sure that this getter is called only once per callback-call.
        return self.log_images_every.match(iteration_count=self.trainer.iteration_count,
                                           epoch_count=self.trainer.epoch_count,
                                           persistent=True)

    def end_of_training_iteration(self, **_):
        # This is very necessary - see comments in the respective property getters.
        log_scalars_now = self.log_scalars_now
        log_images_now = self.log_images_now
        if not log_scalars_now and not log_images_now:
            # Nothing to log, so we won't bother
            return
        # Fetch from trainer
        training_loss = self.trainer.get_state('training_loss')
        training_error = self.trainer.get_state('training_error')
        training_prediction = self.trainer.get_state('training_prediction')
        training_inputs = self.trainer.get_state('training_inputs')
        training_target = self.trainer.get_state('training_target')
        learning_rates = pyu.to_iterable(self.trainer.get_current_learning_rate())

        if log_scalars_now:
            # Extract floats from torch tensors if necessary
            if tu.is_tensor(training_loss):
                training_loss = training_loss.float()[0]
            if tu.is_tensor(training_error):
                training_error = training_error.float()[0]
            # We might have multiple learning rates for multiple groups
            for group_num, learning_rate in enumerate(learning_rates):
                if tu.is_tensor(learning_rate):
                    learning_rate = learning_rate.float()[0]
                self.log_scalar('learning_rate_group_{}'.format(group_num),
                                learning_rate, self.trainer.iteration_count)
            self.log_scalar('training_error', training_error, self.trainer.iteration_count)
            self.log_scalar('training_loss', training_loss, self.trainer.iteration_count)

        if log_images_now:
            # Extract images and log if possible
            # Training prediction
            if pyu.is_maybe_list_of(tu.is_image_or_volume_tensor)(training_prediction):
                self.log_image_or_volume_batch('training_prediction', training_prediction)
            # Training inputs
            if pyu.is_maybe_list_of(tu.is_image_or_volume_tensor)(training_inputs):
                self.log_image_or_volume_batch('training_input', training_inputs)
            # Training target
            if pyu.is_maybe_list_of(tu.is_image_or_volume_tensor)(training_target):
                self.log_image_or_volume_batch('training_target', training_target)

    def extract_images_from_batch(self, batch):
        # Special case when batch is a list or tuple of batches
        if isinstance(batch, (list, tuple)):
            image_list = []
            for _batch in batch:
                image_list.extend(self.extract_images_from_batch(_batch))
            return image_list
        # `batch` really is a tensor from now on.
        batch_is_image_tensor = tu.is_image_tensor(batch)
        batch_is_volume_tensor = tu.is_volume_tensor(batch)
        assert batch_is_volume_tensor != batch_is_image_tensor, \
            "Batch must either be a image or a volume tensor."
        # Convert to numpy
        batch = batch.float().numpy()
        # Get the indices of the batches we want to send to tensorboard
        batch_indices = self._config.get('image_batch_indices', 'all')
        if batch_indices == 'all':
            batch_indices = list(range(batch.shape[0]))
        elif isinstance(batch_indices, (list, tuple)):
            pass
        elif isinstance(batch_indices, int):
            batch_indices = [batch_indices]
        else:
            raise NotImplementedError
        # Get the indices of the channels we want to send to tensorboard
        channel_indices = self._config.get('image_channel_indices', 'all')
        if channel_indices == 'all':
            channel_indices = list(range(batch.shape[1]))
        elif isinstance(channel_indices, (list, tuple)):
            pass
        elif isinstance(channel_indices, int):
            channel_indices = [channel_indices]
        else:
            raise NotImplementedError
        # Extract images from batch
        if batch_is_image_tensor:
            image_list = [image
                          for instance_num, instance in enumerate(batch)
                          for channel_num, image in enumerate(instance)
                          if instance_num in batch_indices and channel_num in channel_indices]
        else:
            assert batch_is_volume_tensor
            # Trim away along the z axis
            z_indices = self._config.get('volume_z_indices', 'mid')
            if z_indices == 'all':
                z_indices = list(range(batch.shape[2]))
            elif z_indices == 'mid':
                z_indices = [batch.shape[2] // 2]
            elif isinstance(z_indices, (list, tuple)):
                pass
            elif isinstance(z_indices, int):
                z_indices = [z_indices]
            else:
                raise NotImplementedError
            # I'm going to hell for this.
            image_list = [image
                          for instance_num, instance in enumerate(batch)
                          for channel_num, volume in enumerate(instance)
                          for slice_num, image in enumerate(volume)
                          if instance_num in batch_indices and
                          channel_num in channel_indices and
                          slice_num in z_indices]
        # Done.
        return image_list

    def log_image_or_volume_batch(self, tag, batch, step=None):
        assert pyu.is_maybe_list_of(tu.is_image_or_volume_tensor)(batch)
        step = step or self.trainer.iteration_count
        image_list = self.extract_images_from_batch(batch)
        self.log_images(tag, image_list, step)

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
            try:
                # Python 2.7
                s = StringIO()
                toimage(image).save(s, format="png")
            except TypeError:
                # Python 3.X
                s = BytesIO()
                toimage(image).save(s, format="png")
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

    def get_config(self):
        # Apparently, some SwigPyObject objects cannot be pickled - so we need to build the
        # writer on the fly.
        config = super(TensorboardLogger, self).get_config()
        config.pop('_writer')
        return config
