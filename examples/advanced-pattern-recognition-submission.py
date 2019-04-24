import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import manifold, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import code.CI_utils.CI_utils.data_preparation as data_prep
import code.CI_utils.CI_utils.graphs as graphs
import code.CI_utils.CI_utils.metrics as metrics
import code.CI_utils.CI_utils.networks_classif as nets
import code.CI_utils.CI_utils.networks_classif_train as nets_train
import code.CI_utils.CI_utils.networks_classif_test as nets_test
import code.CI_utils.CI_utils.networks_classif_visualize as nets_visual

import numpy as np
import cv2


# ------ Global Variables ------

CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ------ Functions ------

def get_hog() :
    winSize = (32,32)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = False

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,
                            derivAperture,winSigma,histogramNormType,L2HysThreshold,
                            gammaCorrection,nlevels, signedGradient)
    return hog


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def extract_hog_labels_from_loader(dataloader, hog):

    hog_descr = []
    labels = []

    for input, label in dataloader:

        un_img = data_prep.unnormalize(input[0], 64, 128)
        inputs = un_img.numpy().transpose(1, 2, 0).astype(np.uint8)

        hog_descr.append(hog.compute(inputs))
        labels.append(label)

    return hog_descr, labels


def direct_features(dataloader):

    descr = []
    labels = []

    for input, label in dataloader:

        input = input[0].numpy().transpose(1, 2, 0).reshape(-1)

        descr.append(input)
        labels.append(label)

    return descr, labels


def logisticRegression(x_train, y_train, x_test, y_test, colors=plt.cm.Purples):

    print('Logistic Regression')

    logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial', max_iter=100)
    logreg.fit(x_train, y_train)

    Yhat_train = logreg.predict(x_train)
    Yhat = logreg.predict(x_test)

    #graphs.plot_confusion_matrix(Yhat, y_test, CLASSES, False, '', cmap=colors)

    err_train = metrics.error(y_train, Yhat_train)
    err_test = metrics.error(y_test, Yhat)

    return err_train, err_test


def knn(x_train, y_train, x_test, y_test, colors=plt.cm.Purples):

    print('kNN')

    neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
    neigh.fit(x_train, y_train)

    Yhat_train = neigh.predict(x_train)
    Yhat = neigh.predict(x_test)

    #graphs.plot_confusion_matrix(Yhat, y_test, CLASSES, False, '', cmap=colors)

    err_train = metrics.error(y_train, Yhat_train)
    err_test = metrics.error(y_test, Yhat)

    return err_train, err_test


def randomForest(x_train, y_train, x_test, y_test, colors=plt.cm.Purples):

    print('Random Forest')

    rf = RandomForestClassifier(n_estimators=128)
    rf.fit(x_train, y_train)

    Yhat_train = rf.predict(x_train)
    Yhat = rf.predict(x_test)

    #graphs.plot_confusion_matrix(Yhat, y_test, CLASSES, False, '', cmap=colors)

    err_train = metrics.error(y_train, Yhat_train)
    err_test = metrics.error(y_test, Yhat)

    return err_train, err_test


if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Define datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    # Define train, val, test set:

    _, subset_train = data_prep.split_dataset(trainset, 0.005)      # obtain 0.1% of the dataset
    subset_train, subset_val = data_prep.split_dataset(subset_train, 0.2)
    _, subset_test = data_prep.split_dataset(testset, 0.01)

    val_split = 0.006
    fraction_of_dataset = 0.01

    #val_splits = np.linspace(0.001, 0.2, 20)
    val_splits = [0.08]

    errors_raw = {}
    errors_hog = {}
    no_samples = []

    for i in ('logreg', 'knn', 'rf'):
        errors_raw[i] = [[], []]
        errors_hog[i] = [[], []]

    for val_split in val_splits:

        print('val_split: {}\n'.format(val_split))

        _, subset_train = data_prep.split_dataset_balanced(trainset, val_split)  # obtain 0.1% of the dataset
        _, subset_val = data_prep.split_dataset(subset_train, val_split)
        _, subset_test = data_prep.split_dataset_balanced(testset, 0.1)

        train_sampler, val_sampler = data_prep.createRandomSampler_multiple(subset_train, subset_val)
        test_sampler = data_prep.createRandomSampler(subset_test)

        # Define loaders:
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, sampler=train_sampler,
                                                  shuffle=False, num_workers=2)

        val_loader = torch.utils.data.DataLoader(trainset, batch_size=1, sampler=val_sampler,
                                                 shuffle=False, num_workers=2)

        #test_loader = torch.utils.data.DataLoader(testset, batch_size=32, sampler=test_sampler,
        #                                          shuffle=False, num_workers=2)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=32,
                                                  shuffle=False, num_workers=2)

        print('\nDataset sizes: Train: {}, Val: {}, Test: {}'.
              format(len(subset_train), len(subset_val), len(subset_test)))

        no_samples.append(len(subset_train))

        # 1. --- Initial Representation for the dataset [Features, Dissimilarities]

        # 1.a - Pixelwise value

        # --- Train/Val
        direct_feats, train_lbls = direct_features(train_loader)
        direct_feats = np.array(direct_feats).squeeze()
        train_lbls = np.array(train_lbls)

        #graphs.tSNE_visualization(direct_feats, train_lbls, '', CLASSES, False)

        # --- Test
        direct_feats_t, test_lbls = direct_features(test_loader)
        direct_feats_t = np.array(direct_feats_t).squeeze()
        test_lbls = np.array(test_lbls)

        # 1.b - Histogram of Gradients representation
        hog = get_hog()
        train_hog, train_lbls = extract_hog_labels_from_loader(train_loader, hog)
        train_hog = np.array(train_hog).squeeze()
        train_lbls = np.array(train_lbls)

        #graphs.tSNE_visualization(train_hog, train_lbls, '', CLASSES, True)

        # --- Test
        test_hog, test_lbls = extract_hog_labels_from_loader(test_loader, hog)
        test_hog = np.array(test_hog).squeeze()
        test_lbls = np.array(test_lbls)

        # 2. --- Study three different classifiers [Learning Curves, Confusion Matrices] ---

        # --- Logistic Regression

        # Pixelwise values
        print('\t raw')
        err_train, err_test = logisticRegression(direct_feats, train_lbls, direct_feats_t,
                                                 test_lbls, plt.cm.Blues)
        errors_raw['logreg'][0].append(err_train)
        errors_raw['logreg'][1].append(err_test)

        # HOG features
        print('\t hog')
        err_train, err_test = logisticRegression(train_hog, train_lbls, test_hog, test_lbls)
        errors_hog['logreg'][0].append(err_train)
        errors_hog['logreg'][1].append(err_test)

        # --- k-NN Classifier

        # Pixelwise values
        print('\t raw')
        err_train, err_test = knn(direct_feats, train_lbls, direct_feats_t, test_lbls,
                                  plt.cm.Blues)
        errors_raw['knn'][0].append(err_train)
        errors_raw['knn'][1].append(err_test)

        # HOG features
        print('\t hog')
        err_train, err_test = knn(train_hog, train_lbls, test_hog, test_lbls)
        errors_hog['knn'][0].append(err_train)
        errors_hog['knn'][1].append(err_test)

        # --- Random Forest (512)

        # Pixelwise values
        print('\t raw')
        err_train, err_test = randomForest(direct_feats, train_lbls, direct_feats_t,
                                           test_lbls, plt.cm.Blues)
        errors_raw['rf'][0].append(err_train)
        errors_raw['rf'][1].append(err_test)

        # HOG features
        print('\t hog')
        err_train, err_test = randomForest(train_hog, train_lbls, test_hog, test_lbls)
        errors_hog['rf'][0].append(err_train)
        errors_hog['rf'][1].append(err_test)

        if val_split == 0.08:

            # Comparison with neural network

            conv_net = nets.VGG_Small(3, len(CLASSES))

            subset_val, subset_train = data_prep.split_dataset_balanced(trainset, val_split)
            print('CNN: length train-set: {} and val-set: {}'.format(len(subset_train), len(subset_val)))

            # Define loaders:
            dataloaders = {}
            dataloaders['train'] = torch.utils.data.DataLoader(trainset, batch_size=32, sampler=train_sampler,
                                                       shuffle=False, num_workers=2)

            dataloaders['validation'] = torch.utils.data.DataLoader(trainset, batch_size=32, sampler=val_sampler,
                                                     shuffle=False, num_workers=2)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(conv_net.parameters(), lr=1e-3)  # , weight_decay=5e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            conv_net.to(device)

            _, av, lv, at, lt = nets_train.train(conv_net, 100, criterion, optimizer,
                                     dataloaders, scheduler, device, "CONV-Model.pth")

            acc, lbls, pred = nets_test.test_model(conv_net, test_loader, device, return_pred=True, load=True,
                                                   path="CONV-Model.pth")

            print('CNN Accuracy: {}'.format(acc))

            graphs.plot_confusion_matrix(pred, lbls, CLASSES, title='', cmap=plt.cm.Oranges)

            encodings, labels = nets_visual.visualize_model(conv_net, dataloaders['train'], device)

            print(np.array(encodings).shape)
            graphs.tSNE_visualization(np.array(encodings), np.array(labels), '', CLASSES, True)

        print('\n\n')

    label = ['Raw', 'HOG']
    for index, error_array in enumerate([errors_raw, errors_hog]):

        plt.figure(dpi=1000)
        colors = {'logreg':'r', 'knn':'g', 'rf':'b'}
        for i in error_array.keys():
            colors_i = [colors[i], colors[i]]
            y_label = 'Classif. Error - '+ label[index] + ' Features'
            graphs.plot_learning_curve(np.array(no_samples), np.array(error_array[i]), '', graph_linecolors=colors_i,
                                       graph_linetypes=['--','-'], graph_markers=['.', 'o'], grid=False,
                                       graph_labels=[i+' - train', i+' - test'], y_label=y_label)
        plt.show()























    #for loader in (train_loader, val_loader, test_loader):

    # Calculate gradient
    #sample = np.float32(sample) / 255.0
    #gx = cv2.Sobel(sample, cv2.CV_32F, 1, 0, ksize=1)
    #gy = cv2.Sobel(sample, cv2.CV_32F, 0, 1, ksize=1)

    #cv2.waitKey()

    #mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    #cv2.imshow('', mag)
    #cv2.imshow('', angle)

    #cv2.waitKey()
    #print(len(hog.compute(sample)))





    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join('%5s' % CLASSES[labels[j]] for j in range(4)))

