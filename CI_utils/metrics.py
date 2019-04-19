from collections import Counter
import numpy as np

def accuracy(labels, predictions, pr=True):
    '''
    Calculates accuracy between predictions and true labels
    :param labels:
    :param predictions:
    :param pr: Bool. Print accuracy value
    :return:
    '''
    assert(len(labels)==len(predictions))

    counter = Counter(labels == predictions)
    acc = counter[True]/float(len(predictions))

    if pr:
        print('Prediction Accuracy: {}'.format(acc))

    return acc


def error(labels, predictions):
    '''
    Calculates error between predictions and true labels
    :param labels:
    :param predictions:
    :return:
    '''
    assert(len(labels)==len(predictions))

    err = 1 - accuracy(labels, predictions, False)

    print('Prediction Error: {}'.format(err))

    return err