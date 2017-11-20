import numpy as np
from .voi import voi
from .arand import adapted_rand
from .base import Metric


class CremiScore(Metric):
    """
    Computes the score used in the Cremi Challenge (www.cremi.org)
    ----------
    Average of VoI-Split, VoI-Merge, RandError.
    """
    def forward(self, prediction, target):
        assert(len(prediction) == len(target))
        segmentation = prediction.cpu().numpy()
        target = target.cpu().numpy()
        return np.mean([sum(CremiScore(segmentation[i], target[i])[0])
                        for i in range(len(prediction))])


def cremi_score(seg, gt, no_seg_ignore=True):
    if no_seg_ignore:
        if 0 in seg:
            seg += 1
    vi_s, vi_m = voi(seg, gt)
    rand = 1. - adapted_rand(seg, gt)[0]
    cs = (vi_s + vi_m + rand) / 3
    return cs, vi_s, vi_m, rand
