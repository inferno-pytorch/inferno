
def implements_sync_primitives(dataset):
    return hasattr(dataset, 'sync_with') and callable(getattr(dataset, 'sync_with'))


def defines_base_sequence(dataset):
    return hasattr(dataset, 'base_sequence') and dataset.base_sequence is not None
