from torch.optim import Adam


class AnnealedAdam(Adam):
    """Implements Adam algorithm with learning rate annealing.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        lr_decay(float, optional): decay learning rate by this factor after every step
            (default: 1.)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, lr_decay=1.):
        params = list(params)
        super(AnnealedAdam, self).__init__(params=params, lr=lr, betas=betas, eps=eps,
                                           weight_decay=weight_decay)
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, lr_decay=lr_decay)
        # We need to initialize the superclass of Adam
        super(Adam, self).__init__(params, defaults=defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # Do an optimization step
        super(AnnealedAdam, self).step(closure=closure)
        # Update learning rate
        for group in self.param_groups:
            group['lr'] *= group['lr_decay']
