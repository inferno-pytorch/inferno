import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
import numpy as np


def generate_random_data(num_samples, shape, num_classes, hardness=0.3, dtype=None):
    """Generate a random dataset with a given hardness and number of classes."""
    dataset_input = np.zeros((num_samples,) + shape, dtype=dtype)
    dataset_target = np.random.randint(num_classes, size=num_samples)
    for sample_num in range(num_samples):
        dataset_input[sample_num] = np.random.normal(loc=dataset_target[sample_num],
                                                     scale=(1 - hardness),
                                                     size=shape)
    return dataset_input, dataset_target


def generate_random_dataset(num_samples, shape, num_classes, hardness=0.3, dtype=None):
    """Generate a random dataset with a given hardness and number of classes."""
    # Generate numpy arrays
    dataset_input, dataset_target = generate_random_data(num_samples, shape, num_classes,
                                                         hardness=hardness, dtype=dtype)
    # Convert to tensor and build dataset
    dataset = TensorDataset(torch.from_numpy(dataset_input),
                            torch.from_numpy(dataset_target))
    return dataset


def generate_random_dataloader(num_samples, shape, num_classes, hardness=0.3, dtype=None,
                               batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
    """Generate a loader with a random dataset of given hardness and number of classes."""
    dataset = generate_random_dataset(num_samples, shape, num_classes, hardness=hardness,
                                      dtype=dtype)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=pin_memory)
    return dataloader
