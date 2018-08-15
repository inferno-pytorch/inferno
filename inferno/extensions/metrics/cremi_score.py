import numpy as np
from .voi import voi
from .arand import adapted_rand


# TODO build metrics object


def cremi_metrics(seg, gt, no_seg_ignore=True):
    if no_seg_ignore:
        if 0  in seg:
            seg += 1
    vi_s, vi_m = voi(seg, gt)
    rand = 1. - adapted_rand(seg, gt)[0]
    cs = np.sqrt((vi_s + vi_m) * rand)
    return cs, vi_s, vi_m, rand
