import warnings

import torch
from torch import nn

from . import set_similarity_measures, core

__all__ = [
    'RegularizedLoss',
    'RegularizedCrossEntropyLoss',
    'RegularizedBCEWithLogitsLoss',
    'RegularizedBCELoss',
    'RegularizedMSELoss',
    'RegularizedNLLLoss'
]


def collect_losses(module):
    """Collect `_losses` dictionaries from module and children

    :param module: a Module to be searched for losses
    :return: dictionary of loss names to values
    """
    losses = {}

    def _collect(m):
        if hasattr(m, '_losses'):
            for k, v in m._losses.items():
                if k in losses:
                    losses[k] = losses[k] + v
                else:
                    losses[k] = v

    module.apply(_collect)
    return losses


def build_criterion(criterion, *args, **kwargs):
    """Build a criterion

    :param criterion: criterion class, name of criterion class, or instance of criterion
    :param args: args for constructor
    :param kwargs: kwargs for constructor
    :return: instance of criterion
    """
    if isinstance(criterion, str):
        for module in [nn, core, set_similarity_measures]:
            criterion_class = getattr(module, criterion, None)
            if criterion_class is not None:
                break
        assert criterion_class is not None, "Criterion {} not found.".format(criterion)
    elif callable(criterion) and isinstance(criterion, type):
        criterion_class = criterion
    elif isinstance(criterion, torch.nn.Module):
        return criterion
    else:
        raise NotImplementedError
    return criterion_class(*args, **kwargs)


class RegularizedLoss(nn.Module):
    """Wrap a criterion. Collect regularization losses from model and combine with wrapped criterion.
    """

    def __init__(self, criterion, *args, **kwargs):
        super(RegularizedLoss, self).__init__()
        self.criterion = build_criterion(criterion, *args, **kwargs)

    def forward(self, *args, trainer=None, model=None, **kwargs):
        # calculate wrapped loss
        main_loss = self.criterion(*args, **kwargs)

        # If no trainer, we cannot record states
        if trainer is None:
            warnings.warn('No trainer parameter provided. Not logging regularization losses.')
        elif model is None:
            model = trainer.model

        # If no model or trainer, we cannot record states or collect losses
        if model is None:
            warnings.warn('No model or trainer parameter provided. Not calculating regularization losses.')
            regularization_losses = {}
            total_regularization_loss = None
            total_loss = main_loss
        else:
            regularization_losses = collect_losses(model)
            total_regularization_loss = sum(regularization_losses.values())
            total_loss = main_loss + total_regularization_loss

        # Record losses if trainer provided
        if trainer is not None:
            # prefix depending on mode
            if self.training:
                prefix = 'training'
            else:
                prefix = 'validation'
            # main loss
            updates = {'{}_main_loss'.format(prefix): main_loss}
            # total regulariztion loss
            if total_regularization_loss is not None:
                updates['{}_total_regularization_loss'.format(prefix)] = total_regularization_loss
            # detailed regularization losses
            for k, v in regularization_losses.items():
                updates['{}_{}'.format(prefix, k)] = v
            # record state
            trainer.update_state_from_dictionary(updates)

        return total_loss


# Convenience wrappers for common losses
class RegularizedCrossEntropyLoss(RegularizedLoss):
    def __init__(self, *args, **kwargs):
        super(RegularizedCrossEntropyLoss, self).__init__(nn.CrossEntropyLoss, *args, **kwargs)


class RegularizedBCEWithLogitsLoss(RegularizedLoss):
    def __init__(self, *args, **kwargs):
        super(RegularizedBCEWithLogitsLoss, self).__init__(nn.BCEWithLogitsLoss, *args, **kwargs)


class RegularizedBCELoss(RegularizedLoss):
    def __init__(self, *args, **kwargs):
        super(RegularizedBCELoss, self).__init__(nn.BCELoss, *args, **kwargs)


class RegularizedMSELoss(RegularizedLoss):
    def __init__(self, *args, **kwargs):
        super(RegularizedMSELoss, self).__init__(nn.MSELoss, *args, **kwargs)


class RegularizedNLLLoss(RegularizedLoss):
    def __init__(self, *args, **kwargs):
        super(RegularizedNLLLoss, self).__init__(nn.NLLLoss, *args, **kwargs)
