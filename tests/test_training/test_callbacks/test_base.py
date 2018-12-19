import unittest
import torch
from inferno.trainers.callbacks.base import Callback, CallbackEngine
from inferno.trainers.basic import Trainer
from os.path import join, dirname, exists
from os import makedirs
from shutil import rmtree


class DummyCallback(Callback):
    def end_of_training_iteration(self, **_):
        assert self.trainer is not None


class WrongDummyCallback(Callback):
    def end_of_iteration(self):
        pass


class CallbackMechTest(unittest.TestCase):
    ROOT_DIR = join(dirname(__file__), 'root')

    def setUp(self):
        makedirs(self.ROOT_DIR, exist_ok=True)

    def tearDown(self):
        if exists(self.ROOT_DIR):
            rmtree(self.ROOT_DIR)

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

    def test_instance_registry(self):
        class Foo(Callback):
            pass

        class Bar(Callback):
            pass

        foo = Foo()
        bar = Bar()
        self.assertIs(foo.get_instances(), foo)
        self.assertIs(bar.get_instances(), bar)
        foo2 = Foo()
        self.assertSequenceEqual(foo2.get_instances(), [foo, foo2])
        self.assertIs(bar.get_instances(), bar)

if __name__ == '__main__':
    unittest.main()
