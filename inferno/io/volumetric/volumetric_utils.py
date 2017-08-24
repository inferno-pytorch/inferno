import random
import itertools as it


# This code is legacy af, don't judge
# Define a sliding window iterator (this time, more readable than a wannabe one-liner)
def slidingwindowslices(shape, nhoodsize, stride=1, ds=1, window=None, ignoreborder=True,
                        shuffle=True, rngseed=None,
                        startmins=None, startmaxs=None, dataslice=None):
    """
    Returns a generator yielding (shuffled) sliding window slice objects.
    :type shape: int or list of int
    :param shape: Shape of the input data
    :type nhoodsize: int or list of int
    :param nhoodsize: Window size of the sliding window.
    :type stride: int or list of int
    :param stride: Stride of the sliding window.
    :type shuffle: bool
    :param shuffle: Whether to shuffle the iterator.
    """

    # Determine dimensionality of the data
    datadim = len(shape)

    # Parse window
    if window is None:
        window = ['x'] * datadim
    else:
        assert len(window) == datadim, \
            "Window must have the same length as the number of data dimensions."

    # Parse nhoodsize and stride
    nhoodsize = [nhoodsize, ] * datadim if isinstance(nhoodsize, int) else nhoodsize
    stride = [stride, ] * datadim if isinstance(stride, int) else stride
    ds = [ds, ] * datadim if isinstance(ds, int) else ds

    # Seed RNG if a seed is provided
    if rngseed is not None:
        random.seed(rngseed)

    # Define a function that gets a 1D slice
    def _1Dwindow(startmin, startmax, nhoodsize, stride, ds, seqsize, shuffle):
        starts = range(startmin, startmax + 1, stride)

        if ignoreborder:
            slices = [slice(st, st + nhoodsize, ds) for st in starts if st + nhoodsize <= seqsize]
        else:
            slices = [slice(st, ((st + nhoodsize) if st + nhoodsize <= seqsize else None), ds)
                      for st in starts]

        if shuffle:
            random.shuffle(slices)
        return slices

    # Get window start limits
    if dataslice is None:
        startmins = [0, ] * datadim if startmins is None else startmins
        startmaxs = [shp - nhoodsiz for shp, nhoodsiz in zip(shape, nhoodsize)] \
            if startmaxs is None else startmaxs
    else:
        assert len(dataslice) == datadim, \
            "Dataslice must be a tuple with len = data dimension."
        startmins = [sl.start for sl in dataslice]
        startmaxs = [sl.stop - nhoodsiz for sl, nhoodsiz in zip(dataslice, nhoodsize)]

    def _to_list(x):
        if not isinstance(x, (list, tuple)):
            return list(x)
        else:
            return x

    # The final iterator is going to be a cartesian product of the lists in nslices
    nslices = [_1Dwindow(startmin, startmax, nhoodsiz, st, dsample, datalen, shuffle) if windowspec == 'x'
               else [slice(ws, ws + 1) for ws in _to_list(windowspec)]
               for startmin, startmax, datalen, nhoodsiz, st, windowspec, dsample in zip(startmins, startmaxs, shape,
                                                                                nhoodsize, stride, window, ds)]

    return it.product(*nslices)


def parse_data_slice(data_slice):
    """Parse a dataslice as a list of slice objects."""
    if data_slice is None:
        return data_slice
    elif isinstance(data_slice, (list, tuple)) and \
            all([isinstance(_slice, slice) for _slice in data_slice]):
        return list(data_slice)
    else:
        assert isinstance(data_slice, str)
    # Get rid of whitespace
    data_slice = data_slice.replace(' ', '')
    # Split by commas
    dim_slices = data_slice.split(',')
    # Build slice objects
    slices = []
    for dim_slice in dim_slices:
        indices = dim_slice.split(':')
        if len(indices) == 2:
            start, stop, step = indices[0], indices[1], None
        elif len(indices) == 3:
            start, stop, step = indices
        else:
            raise RuntimeError
        # Convert to ints
        start = int(start) if start != '' else None
        stop = int(stop) if stop != '' else None
        step = int(step) if step is not None and step != '' else None
        # Build slices
        slices.append(slice(start, stop, step))
    # Done.
    return slices