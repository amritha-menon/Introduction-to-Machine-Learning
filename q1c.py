import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import *


def read_dataset(filename):
    """
    Reads dataset
    :param filename: Name of file
    :return: Data
    """
    df = pd.read_csv(filename, names=["i1", "i2", "i3", "i4", "op"])
    return df


def softmax(z):
    """
    Computes softmax function
    :param z: Class scores
    :return: softmax
    """
    z -= np.max(z)
    return np.exp(z) / np.sum(np.exp(z))


def J(prob, y, W, N, lam):
    """
    Calculates the cost
    :param prob: softmax
    :param y: Target vector
    :param W: Weight vector
    :param N: number of samples
    :param lam: lambda
    :return: J
    """
    return (-1 / N) * np.sum(y * np.log(prob)) + (lam / 2) * np.sum(W * W)


def T(y, k):
    """
    Calculates one-hot encoding of y
    :param y: Target vector
    :param k: number of classes
    :return: one hot encoding of y
    """
    one_hot = np.zeros((len(y), k))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


def compute_gradient(p, y, X, N, lam, W):
    """
    Computes derivatives
    :param p: softmax function
    :param y: Target vector
    :param X: Dataset
    :param N:  number of samples
    :param lam: lambda
    :param W: Weight vector
    :return: dW, db
    """
    dW = weight_der(p, y, X, N, lam, W)
    db = bias_der(p, y, N)
    gradient_loss = p - y
    return dW, db


def create_mini_batch(X, y, batch_size):
    """
    Creates mini batches randomly based on a batch size
    :param X: Dataset
    :param y: Target vector
    :param batch_size: Size of batch
    :return: Mini batches
    """
    mini_batches = []
    randomIndices = random.sample(range(X.shape[0]), X.shape[0])
    x = X[randomIndices]
    y = y[randomIndices]
    for i in range(0, x.shape[0], batch_size):
        Xbatch = x[i:i + batch_size]
        ybatch = y[i:i + batch_size]
        mini_batches.append((Xbatch, ybatch))
    return mini_batches


def train(X, y, X_t, y_t):
    """
    Trains the model using softmax regression
    :param y_t: Test target vector
    :param X_t: Test dataset
    :param X: Train Dataset
    :param y: Train Target vector
    :return: Loss and accuracy
    """

    alpha = 1e-4
    N, d = X.shape
    k = 3
    batch_size = 10
    encode_y = T(y, k)
    encode_yt = T(y_t, k)
    W = np.random.random((d, k))
    b = np.random.random((1, k))
    lam = 0.5
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = list()
    maxEpochs = 1500

    for i in range(maxEpochs):
        train_accuracy = 0
        test_accuracy = 0
        mini_batches = create_mini_batch(X, encode_y, batch_size)
        for mini_batch in mini_batches:
            x_mini, y_mini = mini_batch
            f = np.add(np.dot(x_mini, W), b)
            p = softmax(f)
            cost = J(p, y_mini, W, N, lam)
            dW, db = compute_gradient(p, y_mini, x_mini, N, lam, W)
            W -= alpha * dW
            b -= alpha * db
        train_losses.append(cost)
        for j in range(len(p)):
            if np.argmax(p[j]) == np.argmax(encode_y[j]):
                train_accuracy += 1
        tta = train_accuracy / len(encode_y) * 1000
        train_accuracies.append(tta)

        # test set accuracy and loss calculation
        f_t = np.add(np.dot(X_t, W), b)
        p_t = softmax(f_t)
        test_cost = J(p_t, encode_yt, W, N, lam)
        test_losses.append(test_cost)
        for j in range(len(p_t)):
            if np.argmax(p_t[j]) == np.argmax(encode_yt[j]):
                test_accuracy += 1
            ta = test_accuracy / len(p_t) * 100
            test_accuracies.append(ta)

    return tta, train_accuracies, train_losses, test_accuracies, test_losses, ta


def weight_der(p, y, X, N, lam, W):
    """
   Calculates dj/dW
   :param p: softmax function
   :param y: softmax function
   :param X: dataset
   :param N: number of samples
   :param lam: lambda
   :param W: Weight vector
   :return: Weight derivative
   """
    x = 1 / N * (p - y)
    dW = X.T.dot(x)
    dW += lam * W
    return dW


def bias_der(p, y, N):
    """
    Calculates dj/db
    :param p: softmax function
    :param y: softmax function
    :param N: number of samples
    :return: Bias derivative
    """
    x = np.sum(p - y, axis=0, keepdims=True)
    db = x / N
    return db


def main():
    """
    Main function: Learns maximum entropy model for xor problem
    :return:
    """
    X_train = read_dataset("iris_train.dat")
    X = X_train[["i1", "i2", "i3", "i4"]].values.astype(np.float32)
    y = pd.factorize(X_train['op'])[0]
    X_val = read_dataset("iris_train.dat")
    X_t = X_val[["i1", "i2", "i3", "i4"]].values.astype(np.float32)
    y_t = pd.factorize(X_val['op'])[0]
    tta, train_accuracies, train_losses, test_accuracies, test_losses, ta = train(X, y, X_t, y_t)
    print("Train Accuracy: ", tta, "Test Accuracy: ", ta)

    plt.plot(train_losses, label='Training cost')
    plt.plot(test_losses, label='Testing cost')
    xlabel('Epoch count')
    ylabel('J')
    legend()
    plt.show()

    plt.plot(train_accuracies, label='Training accuracy')
    plt.plot(test_accuracies, label='Testing accuracy')
    xlabel('Epoch count')
    ylabel('Accuracy')
    legend()
    plt.show()


main()
