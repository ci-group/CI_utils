import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data as utils
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

def split_dataset(dataset, validation_split = 0.2):
    '''
    :param dataset: Training Dataset which is to be split.
    :return: Returns indices for training and validation
    '''

    print('Validation Split: ', validation_split)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    print('Sizes are respectively {} and {} samples for train and val respectively'.format(
    len(train_indices), len(val_indices)))

    return train_indices, val_indices

def createRandomSampler_multiple(train_indices, val_indices):
    '''
    :param train_indices: sampled indexes for training
    :param val_indices: sampled indexes for validation
    :return: returns tuple of SubsetRandomSampler (train_sampler, val_sampler)
            (see Torch for more info)
    '''

    train_sampler = createRandomSampler(train_indices)
    val_sampler = createRandomSampler(val_indices)

    return train_sampler, val_sampler

def createRandomSampler(train_indices):
    '''
    :param train_indices: sampled indexes for training
    :param val_indices: sampled indexes for validation
    :return: returns tuple of SubsetRandomSampler (train_sampler, val_sampler)
            (see Torch for more info)
    '''

    train_sampler = SubsetRandomSampler(train_indices)

    return train_sampler


def normalize(samples, std_dev, mean):
    pass


def unnormalize(samples, std_dev, mean):
    '''
    :param samples: normalized samples
    :param std_dev: estimated std_dev
    :param mean:
    :return: Returns unnormalized samples by x = x_norm * std_dev + mean
    '''

    un_samples = samples * std_dev + mean
    return un_samples