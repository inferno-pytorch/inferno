import dill
from datetime import datetime
import os
import shutil

import torch
from numpy import inf
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import data_parallel
from .callbacks.logging.base import Logger
from .callbacks.logging import get_logger

from ..utils import train_utils as tu
from ..utils import python_utils as pyu
from ..utils import torch_utils as thu
from ..extensions import metrics
from ..extensions import optimizers
from ..extensions import criteria
from .callbacks import CallbackEngine
from ..utils.exceptions import assert_, NotSetError, NotTorchModuleError, DeviceError


class Trainer(object):
    """A basic trainer.

    Given a torch model, this class encapsulates the training and validation loops,
    checkpoint creation, logging, CPU <-> GPU transfers and managing data-loaders.

    In addition, this class interacts with the callback engine (found at
    `inferno.trainers.callbacks.base.CallbackEngine`), which manages callbacks at
    certain preset events.

    Notes
    -----
    Logging is implemented as a special callback, in the sense that it's jointly
    managed by the this class and the callback engine. This is primarily because
    general callbacks are not intended to be serializable, but not being able to
    serialize the logger is a nuisance.
    """
    def __init__(self, model=None):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Torch model to bind to.
        """
        # Privates
        # Core
        self._model = None
        self._optimizer = None
        self._criterion = None

        # Metric evaluation
        self._metric = None
        self._evaluate_metric_every = None
        self._metric_evaluation_externally_triggered = False
        self._last_metric_evaluated_at_epoch = 0

        # Logging
        self._logger = None
        self._last_logged = {}
        self._log_directory = {}

        # Data logistics
        self._loaders = {}
        self._loader_iters = {}
        self._loader_specs = {}

        # Iteration and epoch book-keeping
        self._iteration_count = 0
        self._epoch_count = 0
        self._batch_count = 0

        # GPU and dtype business
        self._use_cuda = False
        self._dtype = 'float'
        self._devices = None
        self._base_device_ordinal = None

        # Validation
        self._save_at_best_validation_score = False
        self._best_validation_score = None
        self._is_iteration_with_best_validation_score = False
        self._validate_every = None
        self._num_validation_iterations = None
        # We should exclude the zero-th epoch from validation
        self._last_validated_at_epoch = 0
        self._last_validated_at_iteration = 0
        # This is to allow a callback to trigger a validation by setting
        # trainer.validate_now = True
        self._validation_externally_triggered = False

        # Checkpointing
        self._save_every = None
        self._save_to_directory = None
        # Defaults for file names
        self._checkpoint_filename = 'checkpoint.pytorch'
        self._best_checkpoint_filename = 'best_checkpoint.pytorch'

        # Nothing to save at epoch 0
        self._last_saved_at_epoch = 0
        # This is to allow a callback to trigger a save by setting trainer.save_now = True
        self._save_externally_triggered = False

        # Stopping conditions
        self._max_num_iterations = None
        self._max_num_epochs = None

        # Callbacks and states
        self._callback_engine = CallbackEngine().bind_trainer(self)
        self._state = {}

        # Public
        if model is not None:
            self.model = model

    @property
    def callbacks(self):
        """Gets the callback engine."""
        return self._callback_engine

    def register_callback(self, callback, trigger='auto', **callback_kwargs):
        """
        Registers a callback with the internal callback engine.

        Parameters
        ----------
        callback : type or callable
            Callback to register.
        trigger : str
            Specify the event that triggers the callback. Leave at 'auto' to have the
            callback-engine figure out the triggers. See
            `inferno.training.callbacks.base.CallbackEngine` documentation for more on this.
        callback_kwargs : dict
            If `callback` is a type, initialize an instance with these keywords to the
            __init__ method.
        Returns
        -------
        Trainer
            self.
        """
        if isinstance(callback, type):
            callback = callback(**callback_kwargs)
        self._callback_engine.register_callback(callback, trigger=trigger)
        return self

    @property
    def model(self):
        """Gets the model."""
        assert_(self._model is not None, "Model is not defined yet.", NotSetError)
        return self._model

    @model.setter
    def model(self, value):
        self.bind_model(value)

    def bind_model(self, model):
        """
        Binds a model to the trainer. Equivalent to setting model.

        Parameters
        ----------
        model : torch.nn.Module
            Model to bind.

        Returns
        -------
        Trainer
            self.
        """
        assert_(isinstance(model, torch.nn.Module),
                "Model must be a torch.nn.Module.",
                NotTorchModuleError)
        self._model = model
        # Transfer model to GPU if required
        if self._use_cuda:
            self._model.cuda()
        return self

    @property
    def model_is_defined(self):
        return self._model is not None

    @property
    def optimizer(self):
        """Gets the optimizer."""
        assert_(self._optimizer is not None, "Optimizer is not set yet.", NotSetError)
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        if isinstance(value, str) or callable(value):
            self.build_optimizer(value)
        elif isinstance(value, dict):
            self.build_optimizer(**value)
        else:
            raise NotImplementedError

    @property
    def optimizer_is_defined(self):
        return self._optimizer is not None

    def build_optimizer(self, method, param_groups=None, **kwargs):
        """
        Builds the optimizer for training.

        Parameters
        ----------
        method : str or callable or torch.optim.Optimizer
            Name of the optimizer when str, handle to the optimizer class when callable,
            or a torch.optim.Optimizer instance. If a name is provided, this method looks
            for the optimizer in `torch.optim` module first and in
            inferno.extensions.optimizers second.
        param_groups : list of dict
            Specifies the parameter group. Defaults to model.parameters() if None.
        kwargs : dict
            Keyword arguments to the optimizer.

        Returns
        -------
        Trainer
            self.

        Raises
        ------
        AssertionError
            if optimizer is not found
        NotImplementedError
            if method is not str or callable.
        """
        if isinstance(method, str):
            optimizer_class = getattr(torch.optim, method, None)
            if optimizer_class is None:
                # Look for optimizer in extensions
                optimizer_class = getattr(optimizers, method, None)
            assert optimizer_class is not None, "Optimizer {} not found.".format(method)
        elif callable(method) and isinstance(method, type):
            optimizer_class = method
        elif isinstance(method, torch.optim.Optimizer):
            self._optimizer = method
            return self
        else:
            raise NotImplementedError
        param_groups = self.model.parameters() if param_groups is None else param_groups
        self._optimizer = optimizer_class(param_groups, **kwargs)
        return self

    @property
    def criterion(self):
        """Gets the loss criterion."""
        assert_(self._criterion is not None, "Criterion is not set yet.", NotSetError)
        return self._criterion

    @criterion.setter
    def criterion(self, value):
        if isinstance(value, str) or callable(value):
            self.build_criterion(value)
        elif isinstance(value, dict):
            self.build_criterion(**value)
        else:
            raise NotImplementedError

    def build_criterion(self, method, **kwargs):
        """
        Builds the loss criterion for training.

        Parameters
        ----------
        method : str or callable or torch.nn.Module
            Name of the criterion when str, criterion class when callable, or a
            torch.nn.Module instance. If a name is provided, this method looks
            for the criterion in `torch.nn`.
        kwargs : dict
            Keyword arguments to the criterion class' constructor if applicable.

        Returns
        -------
        Trainer
            self.

        Raises
        ------
        AssertionError
            if criterion is not found.
        NotImplementedError
            if method is neither a str nor a callable.
        """
        if isinstance(method, str):
            # Look for criteria in torch
            criterion_class = getattr(torch.nn, method, None)
            if criterion_class is None:
                # Look for it in extensions
                criterion_class = getattr(criteria, method, None)
            assert criterion_class is not None, "Criterion {} not found.".format(method)
        elif callable(method) and isinstance(method, type):
            criterion_class = method
        elif isinstance(method, torch.nn.Module):
            self._criterion = method
            return self
        else:
            raise NotImplementedError
        self._criterion = criterion_class(**kwargs)
        # Transfer criterion to GPU if required. This is necessary for e.g. weighted loss,
        # where the weight is registered as a buffer.
        # The criterion is to be cuda'ed only if the model is on CUDA (self._use_cuda) and
        # the base_device is not CPU (ordinal -1).
        if hasattr(self, '_base_device_ordinal'):
            # This is to not break old checkpoints
            base_device_ordinal = self._base_device_ordinal
        else:
            base_device_ordinal = None
        if self._use_cuda and base_device_ordinal != 1:
            self._criterion.cuda()
        return self

    @property
    def criterion_is_defined(self):
        return self._criterion is not None

    @property
    def metric(self):
        """Gets the evaluation metric."""
        assert_(self._metric is not None, "Metric is not set yet.", NotSetError)
        return self._metric

    @metric.setter
    def metric(self, value):
        if callable(value) or isinstance(value, str):
            self.build_metric(value)
        else:
            raise NotImplementedError

    @property
    def evaluating_metric_every(self):
        return self._evaluate_metric_every

    def evaluate_metric_every(self, frequency):
        """
        Set frequency of metric evaluation __during training__ (and not during validation).

        Parameters
        ----------
        frequency : inferno.utils.train_utils.Frequency or str or tuple or list or int
            Metric evaluation frequency. If str, it could be (say) '10 iterations' or '1 epoch'.
            If tuple (or list), it could be (10, 'iterations') or (1, 'epoch'). If int
            (say 10), it's interpreted as (10, 'iterations').

        Returns
        -------
        Trainer
            self
        """
        self._evaluate_metric_every = tu.Frequency.build_from(frequency, priority='iterations')
        assert self._evaluate_metric_every.is_consistent
        return self

    @property
    def evaluate_metric_now(self):
        if self._metric_evaluation_externally_triggered:
            # Reset trigger
            self._metric_evaluation_externally_triggered = False
            return True
        elif self._evaluate_metric_every is None:
            # By default, evaluate metric every time
            return True
        elif self._evaluate_metric_every is not None and self._evaluate_metric_every.by_epoch:
            # Don't evaluate if we've done so already this epoch
            if self._last_metric_evaluated_at_epoch == self._epoch_count:
                return False
            else:
                # If we haven't evaluated this epoch, check if we should
                return self._evaluate_metric_every.match(epoch_count=self._epoch_count)
        else:
            # This is reached when evaluate_metric_every is defined and matching by
            # iteration count
            return self._evaluate_metric_every.match(iteration_count=self._iteration_count)

    @evaluate_metric_now.setter
    def evaluate_metric_now(self, value):
        self._metric_evaluation_externally_triggered = bool(value)

    def build_metric(self, method, **kwargs):
        """
        Builds the metric for evaluation.

        Parameters
        ----------
        method : callable or str
            Name of the metric when string, metric class or a callable object
            when callable. If a name is provided, this method looks for the metric in
            `inferno.extensions.metrics`.

        kwargs : dict
            Keyword arguments to the metric class' constructor, if applicable.

        Returns
        -------
        Trainer
            self.

        Raises
        ------
        AssertionError: if the metric is not found.
        """
        if callable(method):
            if isinstance(method, type):
                self._metric = method(**kwargs)
            else:
                self._metric = method
        elif isinstance(method, str):
            assert hasattr(metrics, method), \
                "Could not find the metric '{}'.".format(method)
            self._metric = getattr(metrics, method)()
        else:
            raise NotImplementedError
        return self

    @property
    def metric_is_defined(self):
        """Checks if the metric is defined."""
        return self._metric is not None

    @property
    def train_loader(self):
        assert self._loaders.get('train') is not None
        return self._loaders.get('train')

    @train_loader.setter
    def train_loader(self, value):
        assert isinstance(value, DataLoader)
        self._loaders.update({'train': value})

    @property
    def validate_loader(self):
        assert self._loaders.get('validate') is not None
        return self._loaders.get('validate')

    @validate_loader.setter
    def validate_loader(self, value):
        assert isinstance(value, DataLoader)
        self._loaders.update({'validate': value})

    @property
    def logger(self):
        """Gets the logger."""
        return self._logger

    @logger.setter
    def logger(self, value):
        if isinstance(value, dict):
            self.build_logger(**value)
        else:
            self.build_logger(logger=value)

    @property
    def log_directory(self):
        """Gets the log directory."""
        return self._log_directory

    @log_directory.setter
    def log_directory(self, value):
        """Sets the log directory,"""
        if value is not None:
            self.set_log_directory(value)

    @property
    def saving_every(self):
        """Gets the frequency at which checkpoints are made."""
        return self._save_every

    def save_at_best_validation_score(self, yes=True):
        """Sets whether to save when the validation score is the best seen."""
        self._save_at_best_validation_score = yes
        return self

    @property
    def save_now(self):
        if self._save_externally_triggered:
            # Reset trigger
            self._save_externally_triggered = False
            # Save if externally triggered
            return True
        elif self._is_iteration_with_best_validation_score:
            return self._save_at_best_validation_score
        else:
            # Check if we're saving by epoch
            if self._save_every is not None and self._save_every.by_epoch:
                # Don't save if we've already saved once this epoch
                if self._epoch_count == self._last_saved_at_epoch:
                    return False
                else:
                    # If we haven't saved this epoch, check if we should
                    return self._save_every.match(epoch_count=self._epoch_count)
            else:
                # We're saving by iterations
                return self._save_every is not None and \
                   self._save_every.match(iteration_count=self._iteration_count)

    @save_now.setter
    def save_now(self, value):
        """Can be set to true to trigger a checkpoint creation.."""
        self._save_externally_triggered = bool(value)

    def save_every(self, frequency, to_directory=None,
                   checkpoint_filename=None, best_checkpoint_filename=None):
        """
        Set checkpoint creation frequency.

        Parameters
        ----------
        frequency : inferno.utils.train_utils.Frequency or tuple or str
            Checkpoint creation frequency. Examples: '100 iterations' or '1 epochs'.
        to_directory : str
            Directory where the checkpoints are to be created.
        checkpoint_filename : str
            Name of the checkpoint file.
        best_checkpoint_filename : str
            Name of the best checkpoint file.
        Returns
        -------
        Trainer
            self.
        """
        self._save_every = tu.Frequency.build_from(frequency, priority='iterations')
        assert self._save_every.is_consistent
        self.save_to_directory(to_directory, checkpoint_filename, best_checkpoint_filename)
        return self

    @property
    def save_directory(self):
        return self._save_to_directory

    def save_to_directory(self, to_directory=None, checkpoint_filename=None,
                          best_checkpoint_filename=None):
        if to_directory is not None:
            assert_(isinstance(to_directory, str), exception_type=TypeError)
            if not os.path.exists(to_directory):
                os.makedirs(to_directory)
            else:
                assert os.path.isdir(to_directory)
            self._save_to_directory = to_directory
        if checkpoint_filename is not None:
            assert_(isinstance(checkpoint_filename, str), exception_type=TypeError)
            self._checkpoint_filename = checkpoint_filename
        if best_checkpoint_filename is not None:
            assert_(isinstance(best_checkpoint_filename, str), exception_type=TypeError)
            self._best_checkpoint_filename = best_checkpoint_filename
        return self

    @property
    def validating_every(self):
        return self._validate_every

    @property
    def validate_now(self):
        if self._validation_externally_triggered:
            # Reset trigger
            self._validation_externally_triggered = False
            return True
        elif self._validate_every is not None and self._validate_every.by_epoch:
            # Don't validate if we've done so already this epoch
            if self._last_validated_at_epoch == self._epoch_count:
                return False
            else:
                # If we haven't validated this epoch, check if we should
                return self._validate_every.match(epoch_count=self._epoch_count,
                                                  match_zero=False)
        else:
            # Don't validate if we've done once already this iteration
            if self._last_validated_at_iteration == self._iteration_count:
                return False
            else:
                # If we haven't validated this iteration, check if we should. The `match_zero` is
                # redundant, but we'll leave it on anyway.
                return self._validate_every is not None and \
                       self._validate_every.match(iteration_count=self._iteration_count,
                                                  match_zero=False)

    @validate_now.setter
    def validate_now(self, value):
        self._validation_externally_triggered = bool(value)

    def validate_every(self, frequency, for_num_iterations=None):
        """
        Set validation frequency.

        Parameters
        ----------
        frequency : inferno.utils.train_utils.Frequency or str or tuple or list or int
            Validation frequency. If str, it could be (say) '10 iterations' or '1 epoch'.
            If tuple (or list), it could be (10, 'iterations') or (1, 'epoch'). If int
            (say 10), it's interpreted as (10, 'iterations').
        for_num_iterations : int
            Number of iterations to validate for. If not set, the model is validated on
            the entire dataset (i.e. till the data loader is exhausted).

        Returns
        -------
        Trainer
            self
        """
        self._validate_every = tu.Frequency.build_from(frequency, priority='iterations')
        assert self._validate_every.is_consistent
        self._num_validation_iterations = for_num_iterations
        return self

    @property
    def iteration_count(self):
        return self._iteration_count

    @property
    def epoch_count(self):
        return self._epoch_count

    def build_logger(self, logger=None, log_directory=None, **kwargs):
        """
        Build the logger.

        Parameters
        ----------
        logger : inferno.trainers.callbacks.logging.base.Logger or str or type
            Must either be a Logger object or the name of a logger or the class of a logger.
        log_directory : str
            Path to the directory where the log files are to be stored.
        kwargs : dict
            Keyword arguments to the logger class.

        Returns
        -------
        Trainer
            self
        """
        if isinstance(logger, Logger):
            # Set logger and register with the callback engine.
            self._logger = logger
            self.callbacks.register_callback(self._logger)
        elif callable(logger):
            kwargs.update({'log_directory': log_directory})
            self._logger = logger(**kwargs)
            self.callbacks.register_callback(self._logger)
        elif isinstance(logger, str):
            self._logger = get_logger(logger)(**kwargs)
            self.callbacks.register_callback(self._logger)
        elif logger is None:
            pass
        else:
            raise NotImplementedError

        if log_directory is not None:
            self.set_log_directory(log_directory)
        return self

    def set_log_directory(self, log_directory):
        """
        Set the directory where the log files are to be stored.

        Parameters
        ----------
        log_directory : str
            Directory where the log files are to be stored.

        Returns
        -------
        Trainer
            self
        """
        self._log_directory = log_directory
        if self._logger is not None:
            self._logger.set_log_directory(log_directory)
        return self

    # States that are fetched dynamically from the trainer object via properties are
    # dynamic states. Such states can not be updated.
    # The following dictionary maps state keys to the corresponding trainer attribute
    DYNAMIC_STATES = {'learning_rate': 'current_learning_rate'}

    def update_state(self, key, value):
        assert key not in self.DYNAMIC_STATES, \
            "State at key '{}' cannot be updated because it's dynamic.".format(key)
        self._state.update({key: value})
        return self

    def update_state_from_model_state_hooks(self):
        if hasattr(self.model, '_state_hooks'):
            state_hooks = getattr(self.model, '_state_hooks')
            if isinstance(state_hooks, dict):
                # Unwrap variables (or tensors) and
                self._state.update({state_key: thu.unwrap(state)
                                    for state_key, state in state_hooks.items()})

    def get_state(self, key, default=None):
        if key in self.DYNAMIC_STATES:
            return getattr(self, self.DYNAMIC_STATES.get(key), default)
        else:
            return self._state.get(key, default)

    @property
    def current_learning_rate(self):
        return self.get_current_learning_rate()

    def get_current_learning_rate(self):
        """
        Gets the current learning rate.
        Returns
        -------
        list or float
            List of learning rates if there are multiple parameter groups, or a float
            if there's just one.
        """
        learning_rate = [param_group.get('lr', -1.)
                         for param_group in self.optimizer.param_groups]
        learning_rate = [_learning_rate[0] if thu.is_tensor(_learning_rate) else _learning_rate
                         for _learning_rate in learning_rate]
        return pyu.from_iterable(learning_rate)

    def cuda(self, devices=None, base_device=None):
        """
        Train on the GPU.

        Parameters
        ----------
        devices : list
            Specify the ordinals of the devices to use for dataparallel training.

        base_device : {'cpu', 'cuda'}
            When using data-parallel training, specify where the result tensors
            are collected. If 'cuda', the results are collected in `devices[0]`.

        Returns
        -------
        Trainer
            self
        """
        # Validate base_device
        assert_(base_device in [None, 'cpu', 'cuda'],
                "`base_device` must either be 'cpu' or 'cuda', got {} instead."
                .format(base_device),
                DeviceError)
        if isinstance(devices, int) or (isinstance(devices, (list, tuple)) and len(devices) == 1):
            # No data-parallelism, make sure base_device is not CPU
            assert_(base_device != 'cpu',
                    "Without dataparallelism, `base_device` cannot be 'cpu'.",
                    DeviceError)
        self._base_device_ordinal = {None: None, 'cpu': -1, 'cuda': None}.get(base_device)
        # Move model to CUDA
        if self.model_is_defined:
            self.model.cuda()
        # Move criterion to cuda if base device ordinal is not -1 (i.e. CPU)
        # (the criterion is evaluated on the base device)
        if self.criterion_is_defined and self._base_device_ordinal != -1:
            self.criterion.cuda()
        elif self.criterion_is_defined and self._base_device_ordinal == -1:
            # Criterion is evaluated on the CPU, make sure that's where it lives
            self.criterion.cpu()
        self._use_cuda = True
        self._devices = devices
        return self

    def cpu(self):
        """
        Train on the CPU.

        Returns
        -------
        Trainer
            self
        """
        if self.model_is_defined:
            self.model.cpu()
        if self.criterion_is_defined:
            self.criterion.cpu()
        self._use_cuda = False
        self._devices = None
        return self

    def is_cuda(self):
        """Returns whether using GPU for training."""
        return self._use_cuda

    def to_device(self, objects):
        if isinstance(objects, (list, tuple)):
            return type(objects)([self.to_device(_object) for _object in objects])
        else:
            return objects.cuda() if self._use_cuda else objects

    def apply_model(self, *inputs):
        if hasattr(self, '_base_device_ordinal'):
            # This is to not break old checkpoints
            base_device_ordinal = self._base_device_ordinal
        else:
            base_device_ordinal = None
        if self._devices is not None:
            return data_parallel(self.model, inputs, list(self._devices),
                                 output_device=base_device_ordinal)
        else:
            return self.model(*inputs)

    def cast(self, objects):
        if isinstance(objects, (list, tuple)):
            return type(objects)([self.cast(_object) for _object in objects])
        else:
            # Cast only the float types, while leaving the ints alone
            if objects.__class__.__name__ in ['HalfTensor', 'FloatTensor', 'DoubleTensor']:
                cast_fn = getattr(objects, self._dtype, None)
            else:
                cast_fn = None

            if cast_fn is not None:
                return cast_fn()
            else:
                return objects

    def set_precision(self, dtype):
        """
        Set training precision.

        Parameters
        ----------
        dtype : {'double', 'float', 'half'}
            Training precision.

        Returns
        -------
        Trainer
            self
        """
        assert dtype in ['double', 'float', 'half']
        self._dtype = dtype
        self._model = getattr(self._model, dtype)()
        return self

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self.set_precision(value)

    def bind_loader(self, name, loader, num_inputs=None, num_targets=1):
        """
        Bind a data loader to the trainer.

        Parameters
        ----------
        name : {'train', 'validate', 'test'}
            Name of the loader, i.e. what it should be used for.
        loader : torch.utils.data.DataLoader
            DataLoader object.
        num_inputs : int
            Number of input tensors from the `loader`.
        num_targets : int
            Number of target tensors from the `loader`.

        Returns
        -------
        Trainer
            self

        Raises
        ------
        KeyError
            if name is invalid.
        TypeError
            if loader is not a DataLoader instance.
        """
        assert_(name in ['train', 'validate', 'test'],
                "`name` must be one of ['train', 'validate', 'test']. "
                "Got {} instead.".format(name),
                KeyError)
        assert_(isinstance(loader, DataLoader),
                "`loader` must be a DataLoader object. "
                "Got {} instead.".format(type(loader).__name__),
                TypeError)
        # Check to see if the loader is actually new. This should usually be True.
        is_new_loader = loader is not self._loaders.get(name)
        self._loaders.update({name: loader})
        # We also need to account for the case when a loader is being replaced. When this happens,
        # the old DataLoaderIter might still have processes running, which we need to kill.
        if is_new_loader and name in self._loader_iters:
            # This is when the previous loader already has a DataLoaderIter running.
            # The DataLoaderIter implements a __del__ method, which shuts down workers.
            del self._loader_iters[name]
        # Trainers loaded from pickle files might not have '_loader_specs', therefore:
        if not hasattr(self, '_loader_specs'):
            setattr(self, '_loader_specs', {})
        self._loader_specs.update({name: {'num_inputs': num_inputs,
                                          'num_targets': num_targets}})
        return self

    def get_loader_specs(self, name):
        assert name in self._loader_specs.keys(), \
            "Could not find specs about loader '{}'. Valid loader names are: {}" \
                .format(name, set(self._loader_specs.keys()))
        return self._loader_specs.get(name)

    def fetch_next_batch(self, from_loader='train', restart_exhausted_generators=True,
                         update_batch_count=True, update_epoch_count_if_generator_exhausted=True):
        # Check if the iterator is built
        if from_loader not in self._loader_iters:
            self._loader_iters.update({from_loader: self._loaders[from_loader].__iter__()})
        # Try to fetch from iterator
        try:
            # Fetch
            next_batch = next(self._loader_iters[from_loader])
            # Verify
            self.verify_batch(next_batch, from_loader)
            if update_batch_count:
                self._batch_count += 1
            return next_batch
        except StopIteration:
            # This if clause prevents infinite recursion if the loader is empty
            if restart_exhausted_generators:
                self._loader_iters.update({from_loader: self._loaders[from_loader].__iter__()})
                # Update epoch count
                if update_epoch_count_if_generator_exhausted:
                    self.next_epoch()
                return self.fetch_next_batch(from_loader, restart_exhausted_generators=False,
                                             update_batch_count=update_batch_count)
            else:
                raise

    def verify_batch(self, batch, from_loader):
        loader_specs = self.get_loader_specs(from_loader)
        num_inputs = loader_specs.get('num_inputs')
        num_targets = loader_specs.get('num_targets')
        if None not in [num_inputs, num_targets]:
            assert len(batch) == num_inputs + num_targets, \
                "Was expecting a batch with {} (= num_inputs) + {} (= num_targets) tensors, " \
                "got one with {} tensors.".format(num_inputs, num_targets, len(batch))
        if num_inputs is not None:
            assert len(batch) > num_inputs, \
                "Expecting {} inputs, but the batch contains only {} tensors." \
                    .format(num_inputs, len(batch))
        if num_targets is not None:
            assert len(batch) > num_targets, \
                "Expecting {} outputs, but the batch contains only {} tensors." \
                    .format(num_targets, len(batch))
        return batch

    def split_batch(self, batch, from_loader):
        loader_specs = self.get_loader_specs(from_loader)
        num_inputs = loader_specs.get('num_inputs')
        num_targets = loader_specs.get('num_targets')
        assert not (num_targets is None and num_inputs is None), \
            "Can not split batch if both the number of inputs and targets is not known."
        if num_inputs is None:
            # Unknown number of inputs
            inputs, targets = batch[:-num_targets], batch[-num_targets:]
        elif num_targets is None:
            # Unknown number of targets
            inputs, targets = batch[:num_inputs], batch[num_inputs:]
        else:
            # Known number of inputs and targets
            inputs, targets = batch[:num_inputs], batch[-num_targets:]
        return inputs, pyu.from_iterable(targets)

    def restart_generators(self, of_loader=None):
        if of_loader is None:
            of_loader = self._loaders.keys()
        else:
            assert of_loader in self._loaders.keys(), \
                "Key {} not in loaders ({})".format(of_loader, list(self._loaders))
            of_loader = pyu.to_iterable(of_loader)

        self._loader_iters.update({from_loader: self._loaders[from_loader].__iter__()
                                   for from_loader in of_loader})
        return self

    def wrap_batch(self, batch, from_loader=None, requires_grad=False, volatile=False):
        base_device_ordinal = \
            self._base_device_ordinal if hasattr(self, '_base_device_ordinal') else None
        # First, send to the right device
        if base_device_ordinal is None:
            # Both inputs and labels are sent to the device
            batch = self.to_device(batch)
        elif base_device_ordinal == -1:
            # Input batches go to device, while labels remain on the CPU.
            # To start, we need the number of input batches, i.e. from_loader must not be None
            assert_(from_loader is not None,
                    "`from_loader` needs to be specified if base_device_ordinal is -1 "
                    "(i.e. base device for data-parallel training is CPU).",
                    ValueError)
            loader_spec = self._loader_specs.get(from_loader)
            assert_(loader_spec is not None,
                    "No `loader_spec` found for loader key '{}'.".format(from_loader),
                    RuntimeError)
            # Get number of targets
            num_targets = loader_spec['num_targets']
            # Fetch input batches and send'em to device (leave the targets alone)
            inputs = batch[:-num_targets]
            inputs = self.to_device(inputs)
            # Finally, build the batch
            batch = inputs + batch[-num_targets:]
        else:
            raise ValueError("Internal Error: Invalid base_device_ordinal: {}."
                             .format(base_device_ordinal))
        # Cast to the right dtype
        batch = self.cast(batch)
        # Second, wrap as variable
        batch = type(batch)([Variable(_batch, requires_grad=requires_grad, volatile=volatile)
                             for _batch in batch])
        return batch

    def next_iteration(self):
        self._iteration_count += 1

    def next_epoch(self):
        # Callback before the end of epoch
        self.callbacks.call(self.callbacks.END_OF_EPOCH,
                            epoch_count=self._epoch_count,
                            batch_count=self._batch_count,
                            iteration_count=self._iteration_count)
        self._epoch_count += 1
        self._batch_count = 0
        # Callback after the start of epoch
        self.callbacks.call(self.callbacks.BEGIN_OF_EPOCH,
                            epoch_count=self._epoch_count,
                            batch_count=self._batch_count,
                            iteration_count=self._iteration_count)

    def stop_fitting(self, max_num_iterations=None, max_num_epochs=None):
        # First priority to iteration count
        if max_num_iterations is not None or max_num_epochs is None:
            max_num_iterations = \
                self._max_num_iterations if max_num_iterations is None else max_num_iterations
            assert_(max_num_iterations is not None,
                    "Neither max_num_iterations nor max_num_epochs was set.",
                    RuntimeError)
            return self._iteration_count >= max_num_iterations
        else:
            # max_num_epochs is specified. It could be 'auto', in which case we read from the
            # class attribute
            max_num_epochs = self._max_num_epochs \
                if isinstance(max_num_epochs, str) and max_num_epochs.lower() == 'auto' \
                else max_num_epochs
            return self._epoch_count >= max_num_epochs

    INF_STRINGS = {'inf', 'infinity', 'infty'}

    def set_max_num_iterations(self, max_num_iterations):
        """
        Set the maximum number of training iterations.

        Parameters
        ----------
        max_num_iterations : int or float or str
            Maximum number of training iterations. If float, it should equal numpy.inf.
            If str, it should be one of {'inf', 'infinity', 'infty'}.

        Returns
        -------
        Trainer
            self
        """
        max_num_iterations = \
            inf if max_num_iterations in self.INF_STRINGS else max_num_iterations
        # Validate type
        assert_(isinstance(max_num_iterations, int) or max_num_iterations == inf,
                "max_num_iterations must be an integer or numpy.inf, got {} instead."
                .format(type(max_num_iterations).__name__),
                TypeError)
        self._max_num_iterations = max_num_iterations
        return self

    def set_max_num_epochs(self, max_num_epochs):
        """
        Set the maximum number of training epochs.

        Parameters
        ----------
        max_num_epochs : int or float or str
            Maximum number of training epochs. If float, it should equal numpy.inf.
            If str, it should be one of {'inf', 'infinity', 'infty'}.

        Returns
        -------
        Trainer
            self
        """
        max_num_epochs = inf if max_num_epochs in self.INF_STRINGS else max_num_epochs
        assert_(isinstance(max_num_epochs, int) or max_num_epochs == inf,
                "max_num_epochs must be an integer or numpy.inf, got {} instead."
                .format(type(max_num_epochs).__name__),
                TypeError)
        self._max_num_epochs = max_num_epochs
        return self

    def fit(self, max_num_iterations=None, max_num_epochs=None):
        """
        Fit model.

        Parameters
        ----------
        max_num_iterations : int or float or str
            (Optional) Maximum number of training iterations. Overrides the value set by
            `Trainer.set_max_num_iterations`. If float, it should equal numpy.inf.
            If str, it should be one of {'inf', 'infinity', 'infty'}.
        max_num_epochs : int or float or str
            (Optional) Maximum number of training epochs. Overrides the value set by
            `Trainer.set_max_num_epochs`. If float, it should equal numpy.inf.
            If str, it should be one of {'inf', 'infinity', 'infty'}.

        Returns
        -------
        Trainer
            self

        """
        # Takes care of:
        #   - dispatching train
        #   - validation
        #   - learning rate scheduling
        #   - saving

        max_num_iterations = inf if max_num_iterations in self.INF_STRINGS else max_num_iterations
        max_num_iterations = self._max_num_iterations if max_num_iterations is None \
            else max_num_iterations

        max_num_epochs = inf if max_num_epochs in self.INF_STRINGS else max_num_epochs
        max_num_epochs = self._max_num_epochs if max_num_epochs is None else max_num_epochs

        self.callbacks.call(self.callbacks.BEGIN_OF_FIT,
                            max_num_iterations=max_num_iterations,
                            max_num_epochs=max_num_epochs)

        # Local clock
        run_num = 0
        while True:
            if self.stop_fitting(max_num_iterations, max_num_epochs):
                self.print("Exceeded max number of iterations / epochs, breaking.")
                break
            # Train
            self.train_for(break_callback=lambda *args: self.stop_fitting(max_num_iterations,
                                                                          max_num_epochs))
            # Check if it's time to validate
            if self.validate_now:
                self.print("Validating.")
                self.validate_for()
            # Check if it's time to save
            if self.save_now:
                self.print("Saving.")
                self.save()
            run_num += 1

        # Call callback
        self.callbacks.call(self.callbacks.END_OF_FIT,
                            max_num_iterations=max_num_iterations,
                            max_num_epochs=max_num_epochs,
                            num_runs=run_num)

        return self

    def apply_model_and_loss(self, inputs, target, backward=True):
        # Compute prediction
        prediction = self.apply_model(*inputs)
        # Compute loss
        loss = self.criterion(prediction, target)
        if backward:
            # Backprop if required
            loss.backward()
        return prediction, loss

    def train_for(self, num_iterations=None, break_callback=None):
        # Switch model to train mode
        self.model.train()
        # Call callback
        self.callbacks.call(self.callbacks.BEGIN_OF_TRAINING_RUN,
                            num_iterations=num_iterations)
        # iteration_num is a local clock. There's the global self._iteration_count that keeps
        # actual track of the number of iterations - this is updated by the call to
        # self.next_iteration().
        iteration_num = 0
        while True:
            if num_iterations is not None and iteration_num > num_iterations:
                self.print("Finished {} iterations. Breaking...".format(num_iterations))
                break
            # Break if break callback asks us to
            if break_callback is not None and break_callback(iteration_num):
                self.print("Breaking on request from callback.")
                break
            self.print("Training iteration {} (batch {} of epoch {})."
                       .format(iteration_num, self._batch_count, self._epoch_count))
            # Call callback
            self.callbacks.call(self.callbacks.BEGIN_OF_TRAINING_ITERATION,
                                iteration_num=iteration_num)
            # Zero out the grads
            self.optimizer.zero_grad()
            # No interrupts while computing - a SIGINT could shoot down the driver if
            # done at the wrong time. Not sure if this has something to do with pinned memory
            with pyu.delayed_keyboard_interrupt():
                # Get batch
                batch = self.fetch_next_batch('train')
                # Send to device and wrap as variable
                batch = self.wrap_batch(batch, from_loader='train')
                # Separate inputs from targets
                inputs, target = self.split_batch(batch, from_loader='train')
                # Apply model, compute loss and backprop
                prediction, loss = self.apply_model_and_loss(inputs, target, backward=True)
            # Compute metric
            if self.metric_is_defined and self.evaluate_metric_now:
                self._last_metric_evaluated_at_epoch = self._epoch_count
                error = self.metric(thu.unwrap(prediction, to_cpu=False),
                                    thu.unwrap(target, to_cpu=False))
                self.update_state('training_error', thu.unwrap(error))
            else:
                error = None
            # Update state from computation
            self.update_state('training_inputs', thu.unwrap(inputs))
            self.update_state('training_target', thu.unwrap(target))
            self.update_state('training_prediction', thu.unwrap(prediction))
            self.update_state('training_loss', thu.unwrap(loss))
            # Update state from model's state hooks
            self.update_state_from_model_state_hooks()
            # Update parameters
            self.optimizer.step()
            # Call callback
            self.callbacks.call(self.callbacks.END_OF_TRAINING_ITERATION,
                                iteration_num=iteration_num)
            # Prepare for next iteration
            self.next_iteration()
            # Break if validating or saving. It's important that the next_iteration() method is
            # called before checking validate_now and save_now - because otherwise, the iteration
            # counter is never updated after the first save and validate, resulting in an infinite
            # save + validate loop.
            if self.validate_now:
                self.print("Breaking to validate.")
                break
            if self.save_now:
                self.print("Breaking to save.")
                break
            iteration_num += 1

        self.callbacks.call(self.callbacks.END_OF_TRAINING_RUN, num_iterations=num_iterations)
        return self

    def validate_for(self, num_iterations=None, loader_name='validate'):
        """
        Validate for a given number of validation (if `num_iterations is not None`)
        or over the entire (validation) data set.

        Parameters
        ----------
        num_iterations : int
            Number of iterations to validate for. To validate on the entire dataset,
            leave this as `None`.
        loader_name : str
            Name of the data loader to use for validation. 'validate' is the obvious default.

        Returns
        -------
        Trainer
            self.
        """
        assert_(loader_name in ['validate', 'test', 'train'],
                "Invalid `loader_name`: {}".format(loader_name),
                ValueError)
        # Average over errors
        validation_error_meter = tu.AverageMeter()
        validation_loss_meter = tu.AverageMeter()
        iteration_num = 0
        num_iterations = \
            self._num_validation_iterations if num_iterations is None else num_iterations

        # Switch to eval mode (e.g. for batchnorm, etc.)
        self.model.eval()

        # Record the epoch we're validating in
        self._last_validated_at_epoch = self._epoch_count
        self._last_validated_at_iteration = self._iteration_count
        self.callbacks.call(self.callbacks.BEGIN_OF_VALIDATION_RUN,
                            num_iterations=num_iterations,
                            last_validated_at_epoch=self._last_validated_at_epoch)

        # If we don't know num_iterations, we're validating the entire dataset - so we might as
        # well restart the loader now
        if num_iterations is None:
            self.restart_generators(loader_name)

        while True:
            if num_iterations is not None and iteration_num > num_iterations:
                break

            self.callbacks.call(self.callbacks.BEGIN_OF_VALIDATION_ITERATION,
                                iteration_num=iteration_num)

            try:
                batch = self.fetch_next_batch(loader_name,
                                              restart_exhausted_generators=
                                              num_iterations is not None,
                                              update_batch_count=False,
                                              update_epoch_count_if_generator_exhausted=False)
            except StopIteration:
                self.print("{} generator exhausted, breaking.".format(loader_name))
                break

            self.print("Validating iteration {}.".format(iteration_num))

            # Delay SIGINTs till after computation
            with pyu.delayed_keyboard_interrupt():
                # Wrap
                batch = self.wrap_batch(batch, from_loader=loader_name, volatile=True)
                # Separate
                inputs, target = self.split_batch(batch, from_loader=loader_name)
                # Apply model, compute loss
                output, loss = self.apply_model_and_loss(inputs, target, backward=False)
            batch_size = target.size(0)
            validation_loss_meter.update(loss.data[0], n=batch_size)
            # Compute validation_error
            if self.metric_is_defined:
                validation_error = self.metric(thu.unwrap(output, to_cpu=False),
                                               thu.unwrap(target, to_cpu=False))
                if torch.is_tensor(validation_error):
                    # Convert to float
                    validation_error = validation_error[0]
                self.update_state('validation_error', thu.unwrap(validation_error))
                validation_error_meter.update(validation_error, n=batch_size)

            self.update_state('validation_inputs', thu.unwrap(inputs))
            self.update_state('validation_target', thu.unwrap(target))
            self.update_state('validation_prediction', thu.unwrap(output))
            self.update_state('validation_loss', thu.unwrap(loss))
            # This is here for legacy reasons and will eventually be deprecated.
            self.update_state('validation_input', self.get_state('validation_inputs'))
            # Update from model's state hooks
            self.update_state_from_model_state_hooks()

            self.callbacks.call(self.callbacks.END_OF_VALIDATION_ITERATION,
                                iteration_num=iteration_num)

            iteration_num += 1

        self.print("Done validating. Logging results...")

        # Report
        self.record_validation_results(
            validation_loss=validation_loss_meter.avg,
            validation_error=(validation_error_meter.avg if self.metric_is_defined else None))

        self.callbacks.call(self.callbacks.END_OF_VALIDATION_RUN,
                            validation_loss_meter=validation_loss_meter,
                            validation_error_meter=
                            validation_error_meter if self.metric_is_defined else None)
        return self

    def record_validation_results(self, validation_loss, validation_error):
        # Update state
        self.update_state('validation_loss_averaged', thu.unwrap(validation_loss))
        if validation_error is not None:
            self.update_state('validation_error_averaged', thu.unwrap(validation_error))
        # Prefer the error metric (if provided). This should be handled with care -
        # validation error should either always not be None, or otherwise.
        validation_score = validation_loss if validation_error is None else validation_error
        # Check if validation error is less than the best so far
        if self._best_validation_score is None or validation_score < self._best_validation_score:
            # Best score so far. The following flag will trigger a save
            self._is_iteration_with_best_validation_score = True
            self._best_validation_score = validation_score

    def get_config(self, exclude_loader=True):
        # Returns a config dictionary, like __getstate__. Except optionally without the
        # data loaders (which might be yuuuuuge if it contains the data)
        config_dict = dict(self.__dict__)
        # Loader iterators can't be pickled
        if '_loader_iters' in config_dict:
            config_dict.update({'_loader_iters': {}})
        if exclude_loader:
            if '_loaders' in config_dict:
                config_dict.update({'_loaders': {}})
        return config_dict

    def set_config(self, config_dict):
        # TODO some sanity checks on config_dict (e.g. whether the model is actually a model, etc)
        self.__dict__.update(config_dict)
        # Rebind trainer to callback engine
        self.callbacks.bind_trainer(self)
        # Have callback engine rebind all callbacks to trainer
        self.callbacks.rebind_trainer_to_all_callbacks()
        return self

    def save(self, exclude_loader=True, stash_best_checkpoint=True):
        # Log the epoch for save_now
        self._last_saved_at_epoch = self._epoch_count

        self.callbacks.call(self.callbacks.BEGIN_OF_SAVE,
                            save_to_directory=self._save_to_directory,
                            epoch_count=self._epoch_count,
                            batch_count=self._batch_count,
                            iteration_count=self._iteration_count,
                            is_iteration_with_best_validation_score=self._is_iteration_with_best_validation_score)

        checkpoint_path = os.path.join(self._save_to_directory,
                                       self._checkpoint_filename)
        best_checkpoint_path = os.path.join(self._save_to_directory,
                                            self._best_checkpoint_filename)

        # Save the state dictionary
        torch.save(self.get_config(exclude_loader=exclude_loader),
                   checkpoint_path,
                   pickle_module=dill)

        self.callbacks.call(self.callbacks.END_OF_SAVE,
                            save_to_directory=self._save_to_directory,
                            checkpoint_path=checkpoint_path,
                            best_checkpoint_path=best_checkpoint_path,
                            epoch_count=self._epoch_count,
                            batch_count=self._batch_count,
                            iteration_count=self._iteration_count,
                            is_iteration_with_best_validation_score=self._is_iteration_with_best_validation_score)

        if self._is_iteration_with_best_validation_score and stash_best_checkpoint:
            # Do the stashin'
            shutil.copyfile(checkpoint_path, best_checkpoint_path)

        # This is required to prevent an infinite save loop?
        self._is_iteration_with_best_validation_score = False
        self.print("Saved to {}.".format(self._save_to_directory))
        return self

    def save_model(self, to_directory=None):
        to_directory = self._save_to_directory if to_directory is None else to_directory
        # Save the state dictionary
        torch.save(self.model,
                   os.path.join(to_directory, 'model.pytorch'),
                   pickle_module=dill)
        return self

    def load(self, from_directory=None, best=False, filename=None):
        """
        Load the trainer from checkpoint.

        Parameters
        ----------
        from_directory : str
            Path to the directory where the checkpoint is located. The filename should be
            'checkpoint.pytorch' if best=False, or 'best_checkpoint.pytorch' if best=True.
        best : bool
            Whether to load the best checkpoint. The filename in `from_directory` should be
            'best_checkpoint.pytorch'.
        filename : str
            Overrides the default filename.

        Returns
        -------
        Trainer
            self
        """
        from_directory = self._save_to_directory if from_directory is None else from_directory
        assert from_directory is not None, "Nowhere to load from."
        # Get file name
        if filename is None:
            filename = self._best_checkpoint_filename if best else self._checkpoint_filename
        # Load the dictionary
        config_dict = torch.load(os.path.join(from_directory, filename),
                                 pickle_module=dill)
        # This is required to prevent an infinite save loop?
        self._is_iteration_with_best_validation_score = False
        # Set config
        self.set_config(config_dict)
        return self

    def load_model(self, from_directory=None, filename=None):
        from_directory = self._save_to_directory if from_directory is None else from_directory
        filename = 'model.pytorch' if filename is None else filename
        # Load the model
        model = torch.load(os.path.join(from_directory, filename), pickle_module=dill)
        # Set model
        self.model = model
        return self

    def load_(self, *args, **kwargs):
        # Here for legacy reasons - use load instead.
        return self.load(*args, **kwargs)

    def print(self, message):
        print("[+][{}] {}".format(str(datetime.now()), message))

    @classmethod
    def build(cls, model=None, **trainer_config):
        """Factory function to build the trainer."""
        # Check if trainer is to be loaded from file
        if trainer_config.get('load_from_checkpoint'):
            # Load checkpoint config
            trainer = cls(model).save_every(**trainer_config.get('checkpoint_config'))
            trainer.load_()
        else:
            trainer = cls(model)
            if 'logger_config' in trainer_config:
                trainer.build_logger(**trainer_config.get('logger_config'))
            if 'criterion_config' in trainer_config:
                trainer.build_criterion(**trainer_config.get('criterion_config'))
            if 'optimizer_config' in trainer_config:
                trainer.build_optimizer(**trainer_config.get('optimizer_config'))
            if 'metric_config' in trainer_config:
                trainer.build_metric(**trainer_config.get('metric_config'))
            if 'checkpoint_config' in trainer_config:
                trainer.save_every(**trainer_config.get('checkpoint_config'))
            if 'validation_config' in trainer_config:
                trainer.validate_every(**trainer_config.get('validation_config'))
            if 'max_num_iterations' in trainer_config:
                trainer.set_max_num_iterations(trainer_config.get('max_num_iterations'))
            if 'max_num_epochs' in trainer_config:
                trainer.set_max_num_epochs(trainer_config.get('max_num_epochs'))
            if trainer_config.get('use_cuda'):
                devices = trainer_config.get('use_cuda').get('devices') \
                    if isinstance(trainer_config.get('use_cuda'), dict) else None
                trainer.cuda(devices=devices)
            if 'training_precision' in trainer_config:
                trainer.set_precision(trainer_config.get('training_precision'))
        return trainer
