import unittest
from inferno.trainers.callbacks.scheduling import ManualLR
from torch import nn
from torch.optim import Adam


class TestSchedulers(unittest.TestCase):

    def test_manual_lr(self):
        class DummyTrainer(object):
            def __init__(self):
                self.iteration_count = 0
                self.epoch_count = 0
                self.optimizer = Adam(nn.Linear(10, 10).parameters(), lr=1.)

        manual_lr = ManualLR([((100, 'iterations'), 0.5),
                              ((200, 'iterations'), 0.5),
                              ((200, 'iterations'), 0.1)])
        trainer = DummyTrainer()
        manual_lr._trainer = trainer

        manual_lr.end_of_training_iteration()
        self.assertEqual(trainer.optimizer.param_groups[0]['lr'], 1.)
        trainer.iteration_count = 100
        manual_lr.end_of_training_iteration()
        self.assertEqual(trainer.optimizer.param_groups[0]['lr'], 0.5)
        trainer.iteration_count = 200
        manual_lr.end_of_training_iteration()
        self.assertEqual(trainer.optimizer.param_groups[0]['lr'], 0.025)
        trainer.iteration_count = 300
        self.assertEqual(trainer.optimizer.param_groups[0]['lr'], 0.025)

if __name__ == '__main__':
    unittest.main()
