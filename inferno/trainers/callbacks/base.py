from ...utils import python_utils as pyu


class CallbackEngine(object):
    """
    Gathers and manages callbacks.

    Callbacks are callables which are to be called by trainers when certain events ('triggers')
    occur. They could be any callable object, but if endowed with a `bind_trainer` method,
    it's called when the callback is registered. It is recommended that callbacks
    (or their `__call__` methods) use the double-star syntax for keyword arguments.
    """
    # Triggers
    BEGIN_OF_FIT = 'begin_of_fit'
    END_OF_FIT = 'end_of_fit'
    BEGIN_OF_TRAINING_RUN = 'begin_of_training_run'
    END_OF_TRAINING_RUN = 'end_of_training_run'
    BEGIN_OF_EPOCH = 'begin_of_epoch'
    END_OF_EPOCH = 'end_of_epoch'
    BEGIN_OF_TRAINING_ITERATION = 'begin_of_training_iteration'
    END_OF_TRAINING_ITERATION = 'end_of_training_iteration'
    BEGIN_OF_VALIDATION_RUN = 'begin_of_validation_run'
    END_OF_VALIDATION_RUN = 'end_of_validation_run'
    BEGIN_OF_VALIDATION_ITERATION = 'begin_of_validation_iteration'
    END_OF_VALIDATION_ITERATION = 'end_of_validation_iteration'

    TRIGGERS = {BEGIN_OF_FIT,
                END_OF_FIT,
                BEGIN_OF_TRAINING_RUN,
                END_OF_TRAINING_RUN,
                BEGIN_OF_EPOCH,
                END_OF_EPOCH,
                BEGIN_OF_TRAINING_ITERATION,
                END_OF_TRAINING_ITERATION,
                BEGIN_OF_VALIDATION_RUN,
                END_OF_VALIDATION_RUN,
                BEGIN_OF_VALIDATION_ITERATION,
                END_OF_VALIDATION_ITERATION}

    def __init__(self):
        self._trainer = None
        self._callback_registry = {trigger: [] for trigger in self.TRIGGERS}
        self._last_known_epoch = None
        self._last_known_iteration = None

    def bind_trainer(self, trainer):
        self._trainer = trainer
        return self

    def register_callback(self, callback, trigger='auto', bind_trainer=True):
        assert callable(callback)
        # Automatic callback registration based on their methods
        if trigger == 'auto':
            for trigger in self.TRIGGERS:
                if pyu.has_callable_attr(callback, trigger):
                    self.register_callback(callback, trigger, bind_trainer)
            return self
        # Validate triggers
        assert trigger in self.TRIGGERS
        # Add to callback registry
        self._callback_registry.get(trigger).append(callback)
        # Register trainer with the callback if required
        if bind_trainer and pyu.has_callable_attr(callback, 'bind_trainer'):
            callback.bind_trainer(self._trainer)
        return self

    def call(self, trigger, **kwargs):
        assert trigger in self.TRIGGERS
        kwargs.update({'trigger': trigger})
        for callback in self._callback_registry.get(trigger):
            callback(**kwargs)


class Callback(object):
    """Recommended (but not required) base class for callbacks."""
    def __init__(self):
        self._trainer = None

    @property
    def trainer(self):
        return self._trainer

    def bind_trainer(self, trainer):
        self._trainer = trainer
        return self

    def __call__(self, **kwargs):
        if 'trigger' in kwargs:
            if hasattr(self, kwargs.get('trigger')) and \
                    callable(getattr(self, kwargs.get('trigger'))):
                getattr(self, kwargs.get('trigger'))(**kwargs)

