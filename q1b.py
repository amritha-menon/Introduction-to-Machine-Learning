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
    df = pd.read_csv(filename, names=["i1", "i2", "op"])
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


def plot_data(X, Y, W, b):
    """
    Plots the decision boundary
    :param X: Dataset
    :param Y: Target vector
    :param W: Weight vector
    :param b: Bias vector
    :return:
    """
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_new, y_new = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))

    Z = predict(np.c_[x_new.ravel(), y_new.ravel()], W, b)
    Z = Z.reshape(x_new.shape)

    fig = plt.figure()
    contour(x_new, y_new, Z, cmap=plt.cm.Blues, alpha=0.8)
    scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=cm.Blues)
    xlim(x_new.min(), x_new.max())
    ylim(y_new.min(), y_new.max())
    plt.show()


def train(X, y):
    """
    Trains the model using softmax regression
    :param X: Dataset
    :param y: Target vector
    :return: Loss and accuracy
    """
    alpha = 1e-4
    N, d = X.shape
    k = 3
    encode_y = T(y, k)
    W = np.random.random((d, k))
    b = np.random.random((1, k))
    f = np.add(np.dot(X, W), b)
    lam = 0.5
    losses = []
    total_acc = 0
    maxEpochs = 1500
    for i in range(maxEpochs):
        accuracy = 0
        p = softmax(f)
        cost = J(p, encode_y, W, N, lam)
        losses.append(cost)
        dW, db = compute_gradient(p, encode_y, X, N, lam, W)
        W -= alpha * dW
        b -= alpha * db
        for j in range(len(X)):
            if np.argmax(p[j]) == np.argmax(encode_y[j]):
                accuracy += 1
        accuracy = accuracy / len(encode_y) * 100
        total_acc += accuracy

    plot_data(X, encode_y, W, b)

    return losses, total_acc / maxEpochs


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


def predict(X, W, b):
    """
    Predicts the output
    :param X: Dataset
    :param W: Weight vector
    :param b: Bias vector
    :return: prediction
    """
    y_pred = np.dot(X, W) + b
    y_pred = np.argmax(y_pred, axis=1)

    return y_pred


def main():
    """
    Main function: Learns maximum entropy model for xor problem
    :return:
    """
    spiral = read_dataset("spiral_train.dat")
    X = spiral[["i1", "i2"]].values.astype(np.float32)
    y = pd.factorize(spiral['op'])[0]
    error, accuracy = train(X, y)
    print("Accuracy", accuracy)


main()
