"""Things that work out of the box. ;)"""

from .camvid import CamVid, get_camvid_loaders
from .cityscapes import Cityscapes, get_cityscapes_loaders
from .cifar import get_cifar10_loaders, get_cifar100_loaders


__all__ = [
    'CamVid','get_camvid_loaders', 'Cityscapes', 'get_cityscapes_loaders',
    'get_cifar10_loaders','get_cifar100_loaders'
]