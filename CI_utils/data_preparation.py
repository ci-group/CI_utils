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

    train_indices = list(set(indices).symmetric_difference(set(val_indices)))

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


def to_NNlabels(labels):
    '''
    :param labels: receives labels with numerical orderings [1, ... , N_max, ..., 2]
    :return: returns numpy array with labels of the form [[1, 0, ...], ... , [0, ..., 1] , ... , [0, 1, ...]]
             (see input form)
    '''
    max_class = max(labels)
    NNlabels = np.zeros((len(labels), max_class), dtype=np.float)

    for i in range(len(labels)):
        NNlabels[i, labels[i]-1] = 1.

    return NNlabels


def dataset_from_dict(dict):
    '''
    Creates a tuple (samples, label) adequate for CNN training.
    :param dict: Dict from which to extract the labels. Structure: {label_i:[samples_class_i]}
    :return: tuple (samples, label) adequate for CNN training.
    '''

    samples = []
    labels = []
    key_mapping = {key:i for i, key in enumerate(dict.keys(),0)}

    [samples.extend(i) for _,i in dict.items()]
    [labels.extend([key_mapping[key]] * len(dict[key])) for key in dict.keys()]

    samples = np.array(samples)
    #labels = to_NNlabels(labels) --- Apparently nn_loss requires hot-coded vector with class indices
    labels = np.array(labels)

    return samples, labels


def create_Dataset_for_classif(data, labels):
    '''
    Creates a Dataset out of a list, which can directly be used by PyTorch.

    :param data:
    :param labels:
    :return: torch.Dataset
    '''

    data = torch.from_numpy(data).transpose(2, 1)

    labels = np.expand_dims(labels, axis=1)
    labels = torch.from_numpy(labels).transpose(2, 1)

    return utils.TensorDataset(data, labels)


def create_Dataset_for_reconstr(data):
    '''
    Creates a Dataset out of a list, which can directly be used by PyTorch. For
    econstruction tasks.

    :param data:
    :return: torch.Dataset
    '''

    data = torch.from_numpy(data).transpose(2, 1)
    dataset = utils.TensorDataset(data, data)

    return utils.TensorDataset(dataset, dataset)


def create_Dataset_for_classif_and_reconstr(data, labels):
    '''
    Creates a Dataset out of a list, which can directly be used by PyTorch. For
    combined classification/reconstruction tasks.

    :param data:
    :param labels:
    :return: torch.Dataset
    '''

    labels = np.expand_dims(labels, axis=1)
    labels = np.expand_dims(labels, axis=2)

    data = torch.from_numpy(data).transpose(2, 1)
    labels_reconstr = torch.from_numpy(np.concatenate((data, labels), axis=1)).transpose(2, 1)

    return utils.TensorDataset(data, labels_reconstr)















































