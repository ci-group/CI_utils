import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as utils
from torch.utils.data.sampler import SubsetRandomSampler


def train(model, epochs, criterion, optimizer, dataloader, scheduler=None, device = 'cuda:0', save_path = None):
    '''
    :param model:
    :param epochs:
    :param criterion:
    :param optimizer:
    :param dataloader:
    :param FLAGS:
    :param scheduler:
    :param device:
    :param save_path:
    :return:
    '''

    val_acc_history = []
    train_acc_history = []
    loss_train_history = []
    loss_val_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 30)

        #if scheduler is not None:
        # scheduler.step()

        # Each epoch has train n validation set
        for phase in ['train', 'validation']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0
            running_corrects = 0
            total = 0

            # Iterate over data.
            for inputs, labels in dataloader[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # FwrdPhase:
                    # Loss = sum final output n aux output (in InceptionNet)
                    # BUT while testing, only final output considered.

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # BwrdPhase:
                    if phase == 'train':

                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Deep copy of the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'validation':
                val_acc_history.append(epoch_acc)
                loss_val_history.append(epoch_loss)

            # TODO: Differenciate between the different kind of schedulers and their position in the train

            if scheduler is not None:
              scheduler.step(epoch_acc)

            if phase == 'train':
                train_acc_history.append(epoch_acc)
                loss_train_history.append(epoch_loss)

        print()

    print('Best Val Acc: {:.4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    return model, val_acc_history, loss_val_history, train_acc_history, loss_train_history