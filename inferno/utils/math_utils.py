

def max_allowed_ds_steps(shape, factor):
    """How often can a shape be down-sampled by a given factor
        such that non of the divisions will give non-integers.

    Args:
        shape (listlike): tensor shape
        factor (integer): downsample factor

    Returns:
        int: maximum allowed downsample operations
    """
    def max_allowed_ds_steps_impl(size, factor):

        current_size = float(size)
        allowed_steps = 0
        while(True):

            new_size = current_size / float(factor)
            if(new_size >=1 and new_size.is_integer()):

                current_size = new_size
                allowed_steps += 1
            else:
                break
        return allowed_steps

    min_steps = float('inf')

    for s in shape:
        min_steps = int(min(min_steps, max_allowed_ds_steps_impl(s, factor)))

    return min_steps
