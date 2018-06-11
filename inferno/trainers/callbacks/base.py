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
    BEGIN_OF_SAVE = 'begin_of_save'
    END_OF_SAVE = 'end_of_save'

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
                END_OF_VALIDATION_ITERATION,
                BEGIN_OF_SAVE,
                END_OF_SAVE}

    def __init__(self):
        self._trainer = None
        self._callback_registry = {trigger: set() for trigger in self.TRIGGERS}
        self._last_known_epoch = None
        self._last_known_iteration = None

    def register_new_trigger(self, trigger_name):
        self.TRIGGERS.add(trigger_name)
        self._callback_registry.update({trigger_name: set()})

    def bind_trainer(self, trainer):
        self._trainer = trainer
        return self

    def unbind_trainer(self):
        self._trainer = None
        return self

    @property
    def trainer_is_bound(self):
        return self._trainer is not None

    def register_callback(self, callback, trigger='auto', bind_trainer=True):
        assert callable(callback)
        # Automatic callback registration based on their methods
        if trigger == 'auto':
            automatic_registration_successful = False
            for trigger in self.TRIGGERS:
                if pyu.has_callable_attr(callback, trigger):
                    automatic_registration_successful = True
                    self.register_callback(callback, trigger, bind_trainer)
            assert automatic_registration_successful, \
                "Callback could not be auto-registered: no triggers recognized."
            return self
        # Validate triggers
        assert trigger in self.TRIGGERS
        # Add to callback registry
        self._callback_registry.get(trigger).add(callback)
        # Register trainer with the callback if required
        bind_trainer_to_callback = self.trainer_is_bound and \
                                   bind_trainer and \
                                   pyu.has_callable_attr(callback, 'bind_trainer')
        if bind_trainer_to_callback:
            callback.bind_trainer(self._trainer)
        return self

    def rebind_trainer_to_all_callbacks(self):
        # FIXME This makes bind_trainer in register_callback reduntant,
        # especially if used by the trainer class, so... deprecate bind_traner.
        for callbacks_at_trigger in self._callback_registry.values():
            for callback in callbacks_at_trigger:
                # Register trainer with the callback if required
                bind_trainer_to_callback = self.trainer_is_bound and \
                                           pyu.has_callable_attr(callback, 'bind_trainer')
                if bind_trainer_to_callback:
                    callback.bind_trainer(self._trainer)

    def call(self, trigger, **kwargs):
        assert trigger in self.TRIGGERS
        kwargs.update({'trigger': trigger})
        for callback in self._callback_registry.get(trigger):
            callback(**kwargs)

    def get_config(self):
        # Pop trainer
        config_dict = dict(self.__dict__)
        config_dict.update({'_trainer': None})
        return config_dict

    def set_config(self, config_dict):
        self.__dict__.update(config_dict)
        return self

    def __getstate__(self):
        return self.get_config()

    def __setstate__(self, state):
        self.set_config(state)


class Callback(object):
    """Recommended (but not required) base class for callbacks."""
    def __init__(self):
        self._trainer = None
        self._debugging = False
        self.register_instance(self)

    @classmethod
    def register_instance(cls, instance):
        if hasattr(cls, '_instance_registry') and instance not in cls._instance_registry:
            cls._instance_registry.append(instance)
        else:
            cls._instance_registry = [instance]

    @classmethod
    def get_instances(cls):
        if hasattr(cls, '_instance_registry'):
            return pyu.from_iterable(cls._instance_registry)
        else:
            return None

    @property
    def trainer(self):
        return self._trainer

    def bind_trainer(self, trainer):
        self._trainer = trainer
        return self

    def unbind_trainer(self):
        self._trainer = None
        return self

    def __call__(self, **kwargs):
        if 'trigger' in kwargs:
            if hasattr(self, kwargs.get('trigger')) and \
                    callable(getattr(self, kwargs.get('trigger'))):
                getattr(self, kwargs.get('trigger'))(**kwargs)

    def get_config(self):
        config_dict = dict(self.__dict__)
        config_dict.update({'_trainer': None})
        return config_dict

    def set_config(self, config_dict):
        self.__dict__.update(config_dict)
        return self

    def __getstate__(self):
        return self.get_config()

    def __setstate__(self, state):
        self.set_config(state)

    def toggle_debug(self):
        self._debugging = not self._debugging
        return self

    def debug_print(self, message):
        if self._debugging:
            self.trainer.console.debug("[{}] {}".format(type(self).__name__, message))
