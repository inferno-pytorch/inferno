# easy support for additional ranger optimizers from
# https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
try:
    from ranger import Ranger, RangerVA, RangerQH
except ImportError:
    Ranger = None
    RangerVA = None
    RangerQH = None
