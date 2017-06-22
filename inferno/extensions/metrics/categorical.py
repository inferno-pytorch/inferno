import torch


class CategoricalError(object):
    def __init__(self, aggregation_mode='mean'):
        assert aggregation_mode in ['mean', 'sum']
        self.aggregation_mode = aggregation_mode

    def forward(self, prediction, target):
        # Check if prediction is binary or not
        is_binary = len(prediction.size()) == 1 or prediction.size(1) == 1

        if len(target.size()) > 1:
            target = target.squeeze(1)
        assert len(target.size()) == 1

        if is_binary:
            # Binary classification
            prediction = prediction > 0.5
            incorrect = prediction.type_as(target).ne(target)
            if self.aggregation_mode == 'mean':
                return incorrect.sum() / incorrect.size(0)
            else:
                return incorrect.sum()
        else:
            # Multiclass classificiation
            _, predicted_class = torch.max(prediction, 1)
            incorrect = predicted_class.squeeze(1).type_as(target).ne(target)
            if self.aggregation_mode == 'mean':
                return incorrect.sum() / incorrect.size(0)
            else:
                return incorrect.sum()

    def __call__(self, *args):
        return self.forward(*args)
