

class CallbackEngine(object):

    # Triggers
    BEGIN_OF_FIT = 'begin_of_fit'
    END_OF_FIT = 'end_of_fit'
    BEGIN_OF_TRAINING = 'begin_of_training'
    END_OF_TRAINING = 'end_of_training'
    BEGIN_OF_EPOCH = 'begin_of_epoch'
    END_OF_EPOCH = 'end_of_epoch'
    BEGIN_OF_TRAINING_ITERATION = 'begin_of_training_iteration'
    END_OF_TRAINING_ITERATION = 'end_of_training_iteration'
    BEGIN_OF_VALIDATION = 'begin_of_validation'
    END_OF_VALIDATION = 'end_of_validation'

    TRIGGERS = {BEGIN_OF_FIT,
                END_OF_FIT,
                BEGIN_OF_TRAINING,
                END_OF_TRAINING,
                BEGIN_OF_EPOCH,
                END_OF_EPOCH,
                BEGIN_OF_TRAINING_ITERATION,
                END_OF_TRAINING_ITERATION,
                BEGIN_OF_VALIDATION,
                END_OF_VALIDATION}

    def __init__(self):
        self._trainer = None
        self._callback_registry = {trigger: [] for trigger in self.TRIGGERS}
        self._last_known_epoch = None
        self._last_known_iteration = None

    def bind_trainer(self, trainer):
        self._trainer = trainer
        return self

    def register_callback(self, callback, trigger):
        assert trigger in self.TRIGGERS
        assert callable(callback)
        # Add to callback registry
        self._callback_registry.get(trigger).append(callback)
        # Register trainer with the callback if required
        if hasattr(callback, 'bind_trainer') and \
                callable(getattr(callback, 'bind_trainer')):
            callback.bind_trainer(self._trainer)
        return self

    def call(self, trigger, *args, **kwargs):
        assert trigger in self.TRIGGERS
        for callback in self._callback_registry.get(trigger):
            callback(*args, **kwargs)
