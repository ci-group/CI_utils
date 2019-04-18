import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_pred, y_true, classes, normalize=False, title='Confusion matrix',
                          sideLabels=False, cmap=plt.cm.Purples):
    '''
    :param cm: NumPy Array -- Confusion Matrix to be shown.
    :param classes: List -- class_names
    :param normalize: bool
    :param title:
    :param sideLabels: If true, the axis labels [trueLabel, predictedLabels] will appear
                       in the image.
    :param cmap: colormaps type. See plt.cm for more information
    :return: Confusion Matrix
    '''

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(dpi=1000)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-45, horizontalalignment = 'left')
    plt.yticks([])
    #plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    if sideLabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    plt.tight_layout()

    plt.show()


def plot_series(x, y, title, x_label='', y_label='', graph_labels=None, graph_markers=None,
                graph_linetypes=None, graph_linecolors=None, integer_thicks = False, grid=False):
    '''
    Plots the data series contained in y vs x.

    :param x: NumPy Array
    :param y: List of NumPy Arrays
    :param title:
    :param x_label:
    :param y_label:
    :param graph_labels: Labels per {x,y} plot pairs.
    :param graph_markers: https://matplotlib.org/api/markers_api.html
    :param graph_linetypes: https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html
    :param graph_linecolors: https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html
    :param integer_thicks: bool - Force integer tick labels
    :param grid: bool
    :return: Returns graph containing y vs x.
    '''

    plt.title(title)

    for i in range(len(y)):
        label = graph_labels[i] if graph_labels else ''
        marker = graph_markers[i] if graph_markers else ''
        line_color = graph_linecolors[i] if graph_linecolors else None
        line_type = graph_linetypes[i] if graph_linetypes else '-'

        plt.plot(x, y[i], marker=marker, color=line_color, linestyle=line_type,label=label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.grid(grid)

    plt.xlim(x.min(), x.max())

    if graph_labels:
        plt.legend()

    plt.tight_layout()

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=integer_thicks))


def plot_acc_vs_epochs(acc_list, title, graph_labels=None, graph_markers=None,
                       graph_linetypes=None, graph_linecolors=None, grid=False):
    '''
    Returns plot of multiple accuracies vs epochs

    :param acc_list: List of NumPy Arrays - Accuracies for multiple classifiers
    :param title:
    :param graph_labels: Labels per {x,y} plot pairs.
    :param graph_markers: https://matplotlib.org/api/markers_api.html
    :param graph_linetypes: https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html
    :param graph_linecolors: https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html

    :param grid: bool
    :return:
    '''

    x = np.linspace(1, len(acc_list[0]), num=len(acc_list[0]))
    plot_series(x, acc_list, title, 'Epoch', 'Classification Accuracy', graph_labels,
                graph_markers, graph_linetypes, graph_linecolors, True, grid)


def plot_loss_vs_epochs(loss_list, title, graph_labels=None, graph_markers=None,
                        graph_linetypes=None, graph_linecolors=None, grid=False):
    '''
    Returns plot of multiple accuracies vs epochs

    :param loss_list: List of NumPy Arrays - Loss for multiple classifiers per epoch
    :param title:
    :param graph_labels: Labels per {x,y} plot pairs.
    :param graph_markers: https://matplotlib.org/api/markers_api.html
    :param graph_linetypes: https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html
    :param graph_linecolors: https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html
    :param grid: bool
    :return:
    '''

    x = np.linspace(1, len(loss_list[0]), num=len(loss_list[0]))
    plot_series(x, loss_list, title, 'Epoch', 'Epoch Loss', graph_labels,
                graph_markers, graph_linetypes, graph_linecolors, True, grid)



if __name__ == '__main__':

    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]

    #conf_matrix_ccm = np.array([[216, 2, 14, 0, 2, 26, 0, 0], [0, 214, 0, 0, 9, 2, 0, 3], [7, 0, 141, 3, 2, 4, 3, 0],
    #                           [3, 1, 5, 165, 0, 5, 11, 18], [8, 16, 5, 0, 215, 24, 2, 4],
    #                           [43, 14, 10, 3, 13, 217, 7, 3],
    #                           [0, 2, 4, 6, 2, 1, 170, 7], [1, 4, 0, 19, 6, 2, 1, 223]])

    class_names = ['coast', 'forest', 'highway']#, 'insidecity', 'mountain', 'opencountry', 'street', 'tallbuilding']

    # Compute confusion matrix
    # cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    color = plt.cm.Purples

    plt.figure()
    plot_confusion_matrix(y_pred, y_true, classes=class_names,
                          title='Confusion matrix - CCM', sideLabels=True)

    #plt.savefig('confMatrix_CCM.png', format='png', dpi=1000)
    #plt.show()

    plt.figure()
    y = [np.array([3,4,5,6,7,9,10]), np.array([4,5,6,7,9,10,1])]
    x = np.linspace(2.0, 3.0, num=len(y[0]))

    plot_series(x, y, 'Example','x_label', 'y_label', ['a', 'b'], ['.', 'o'],
                ['-','--'], ['b', 'c'], integer_thicks=False, grid=True)
    plt.show()

    plt.figure()
    y = [np.array([3, 4, 5, 6, 7, 9, 10])]
    plot_acc_vs_epochs(y, 'Classification Performance', ['classif_a'], grid=True)
    plt.show()

    plt.figure()
    y = [np.array([3, 4, 5, 6, 7, 9, 10])]
    plot_loss_vs_epochs(y, 'Classification Performance', ['classif_a'],grid=True)
    plt.show()


