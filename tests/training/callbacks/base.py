import unittest
import torch
from inferno.trainers.callbacks.base import Callback, CallbackEngine
from inferno.trainers.basic import Trainer
from os.path import join, dirname


class DummyCallback(Callback):
    def end_of_training_iteration(self, **_):
        assert self.trainer is not None


class WrongDummyCallback(Callback):
    def end_of_iteration(self):
        pass


class CallbackMechTest(unittest.TestCase):
    ROOT_DIR = join(dirname(__file__), 'root')

    def test_serialization(self):
        # Build engine and trainer
        callback_engine = CallbackEngine().bind_trainer(Trainer())
        callback_engine.register_callback(DummyCallback())
        # Serialize
        torch.save(callback_engine, join(self.ROOT_DIR, 'callback_engine.pkl'))
        # Unserialize
        callback_engine = torch.load(join(self.ROOT_DIR, 'callback_engine.pkl'))
        # Make sure the trainer is detached
        self.assertIsNone(callback_engine._trainer)
        self.assertIsInstance(next(iter(callback_engine
                                        ._callback_registry
                                        .get('end_of_training_iteration'))),
                              DummyCallback)

    def test_auto_registry(self):
        callback_engine = CallbackEngine().bind_trainer(Trainer())
        callback_engine.register_callback(DummyCallback())
        self.assertIsInstance(next(iter(callback_engine
                                        ._callback_registry
                                        .get('end_of_training_iteration'))),
                              DummyCallback)
        with self.assertRaises(AssertionError):
            callback_engine.register_callback(WrongDummyCallback())


if __name__ == '__main__':
    unittest.main()
