import os
import pickle
import numpy as np


def load_batch(fpath, label_key='labels'):
    with open(fpath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    data = d['data']
    labels = d[label_key]
    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def label2onehot(Yraw):
    N = Yraw.shape[0]
    Y = np.zeros((N, 10))
    Yraw = list(map(int, Yraw))
    Y[range(N), Yraw] = 1
    return Y


def genDev(X, Y):
    N = X.shape[0]
    perm = np.random.permutation(N)
    ind1 = perm[:int(N / 10)]
    ind2 = perm[int(N / 10):]
    Xdev, Xtr = X[ind1, ], X[ind2, ]
    Ydev, Ytr = Y[ind1, ], Y[ind2, ]
    return Xtr, Xdev, Ytr, Ydev


def extract_data():
    num_train_samples = 50000
    X_rawtrain = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    Y_rawtrain = np.empty((num_train_samples,), dtype='uint8')
    path = "CIFAR10/cifar-10-batches-py"

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (X_rawtrain[(i - 1) * 10000: i * 10000, :, :, :],
         Y_rawtrain[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    X_rawtest, Y_rawtest = load_batch(os.path.join(path, 'test_batch'))

    Y_rawtrain = np.reshape(Y_rawtrain, (len(Y_rawtrain), 1))
    Y_rawtest = np.reshape(Y_rawtest, (len(Y_rawtest), 1))
    X_rawtrain = X_rawtrain.transpose(0, 2, 3, 1)  # channel last
    X_rawtest = X_rawtest.transpose(0, 2, 3, 1)  # channel last
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    X_train = X_rawtrain / 255
    Y_train = label2onehot(Y_rawtrain)
    X_test = X_rawtest / 255
    Y_test = label2onehot(Y_rawtest)

    X_train, X_dev, Y_train, Y_dev = genDev(X_train, Y_train)

    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test, class_names
