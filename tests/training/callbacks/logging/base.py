import unittest
from inferno.trainers.callbacks.logging.base import Logger
from inferno.trainers.basic import Trainer
from os.path import join, dirname


class DummyLogger(Logger):
    def end_of_training_iteration(self, **_):
        pass


class TestLogger(unittest.TestCase):
    ROOT = dirname(__file__)

    def test_serialization(self):
        trainer = Trainer()\
            .build_logger(logger=DummyLogger())\
            .save_to_directory(join(self.ROOT, 'saves'))
        trainer.save()
        # Unserialize
        trainer = Trainer().load(from_directory=join(self.ROOT, 'saves'))
        # Check if the loggers are consistent
        logger_from_trainer = trainer._logger
        logger_from_callback_engine = \
            next(iter(trainer.callbacks._callback_registry['end_of_training_iteration']))
        self.assertIs(logger_from_trainer, logger_from_callback_engine)
        self.assertIs(logger_from_callback_engine.trainer, trainer)


if __name__ == '__main__':
    unittest.main()