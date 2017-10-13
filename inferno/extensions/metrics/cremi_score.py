from .voi import voi
from .arand import adapted_rand


# TODO build metrics object


def cremi_metrics(seg, gt):
    vi_s, vi_m = voi(seg, gt)
    rand = adapted_rand(seg, gt)[0]
    cs = (vi_s + vi_m + rand) / 3
    return cs, vi_s, vi_m, rand
