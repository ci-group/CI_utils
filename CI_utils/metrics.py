from collections import Counter
import numpy as np

def accuracy(labels, predictions):
    '''
    Calculates accuracy between predictions and true labels
    :param labels:
    :param predictions:
    :return:
    '''
    assert(len(labels)==len(predictions))

    counter = Counter(labels == predictions)
    acc = counter[True]/float(len(predictions))

    print('Prediction Accuracy: {}'.format(acc))

    return acc