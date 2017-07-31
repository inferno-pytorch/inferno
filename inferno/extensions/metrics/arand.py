import torch
from .base import Metric
import numpy as np
import scipy.sparse as sparse


class ArandScore(Metric):
    """Arand Score, as defined by http://journal.frontiersin.org/article/10.3389/fnana.2015.00142/full#h3"""
    def forward(self, prediction, target):
        assert(len(prediction) == len(target))
        seg = prediction.cpu().numpy()
        targ = target.cpu().numpy()
        return np.mean([adapted_rand(seg[i], targ[i])[0] for i in range(len(prediction))])

class ArandError(ArandScore):
    """Arand Error = 1 - <arand score>"""
    def forward(self, prediction, target):
        return 1.-super(ArandError, self).forward(prediction, target)

# Evaluation code courtesy of Juan Nunez-Iglesias, taken from
# https://github.com/janelia-flyem/gala/blob/master/gala/evaluate.py
def adapted_rand(seg, gt):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]
    Formula is given as 1 - the maximal F-score of the Rand index
    (excluding the zero component of the original labels). Adapted
    from the SNEMI3D MATLAB script, hence the strange style.
    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error
    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)
    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """
    if np.any(seg == 0):
        print('waarning zeros in seg, treat as background')
    if np.any(gt == 0):
        print('waarning zeros in gt, 0 labels will be ignored')

    if np.all(seg == 0) or np.all(gt == 0):
        print('all labels 0,  fake rand this should not be here. Check in segmentation script')
        if all_stats:
            return (0, 0, 1)
        else:
            return 0

    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)

    # mask to foreground in A
    mask = (segA > 0)
    segA = segA[mask]
    segB = segB[mask]
    n = segA.size  # number of nonzero pixels in original segA

    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

    ones_data = np.ones(n)

    p_ij = sparse.csr_matrix((ones_data, (segA.ravel(), segB.ravel())),
                             shape=(n_labels_A, n_labels_B),
                             dtype=np.uint64)

    # In the paper where adapted rand is proposed, they treat each background
    # pixel in segB as a different value (i.e., unique label for each pixel).
    # To do this, we sum them differently than others

    B_nonzero = p_ij[:, 1:]             # ind (label_gt, label_seg), so ignore 0 seg labels
    B_zero = p_ij[:, 0]

    # this is a count
    num_B_zero = B_zero.sum()

    # sum of the joint distribution
    #   separate sum of B>0 and B=0 parts
    sum_p_ij = (B_nonzero).power(2).sum() + num_B_zero

    # these are marginal probabilities
    a_i = p_ij.sum(1)           # sum over all seg labels overlapping one gt label (except 0 labels)
    b_i = B_nonzero.sum(0)

    sum_a = np.power(a_i, 2).sum()
    sum_b = np.power(b_i, 2).sum() + num_B_zero

    precision = float(sum_p_ij) / sum_b
    recall = float(sum_p_ij) / sum_a

    fScore = 2.0 * precision * recall / (precision + recall)

    return [fScore, precision, recall]
