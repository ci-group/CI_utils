import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.utils.data as utils
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import random


def split_dataset(dataset, validation_split = 0.2):
    '''
    splits dataset into training and validation subsets randomly.

    NOTE!: number of samples can differ per class.

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


def split_dataset_balanced(dataset, validation_split=0.2):
    '''
    splits dataset into training and validation subsets in a balanced manner.
    i.e. len(class_i)=len(class_j) \forall {i,j} \in Classes.

    :param dataset: Training Dataset which is to be split.
    :return: Returns indices for training and validation
    '''

    print('Validation Split: ', validation_split)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    labels = dataset.targets

    indices_per_label = [[] for i in range(len(np.unique(labels)))]
    [indices_per_label[labels[i]].append(indices[i]) for i in range(len(indices))]

    split = int(np.floor(validation_split * len(indices_per_label[0])))

    val_indices = []

    for i in indices_per_label:
        val_indices.extend(random.sample(i, split))

    train_indices = set(indices).symmetric_difference(set(val_indices))

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