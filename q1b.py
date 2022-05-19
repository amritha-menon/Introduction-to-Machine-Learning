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
    alpha = 0.0001
    N, d = X.shape

    k = 3
    W1 = np.random.random((d, k))
    W2 = np.random.random((d, k))
    b1 = np.random.random((1, k))
    b2 = np.random.random((1, k))
    encode_y = T(y, k)

    lam = 0.5
    losses = []
    accs = []
    total_acc = 0
    maxEpochs = 8000

    for i in range(maxEpochs):

        f1 = np.add(np.dot(X, W1), b1)
        f2 = np.add(np.dot(X, W2), b2)
        p1 = softmax(f1)
        p2 = softmax(f2)
        print((encode_y - p2).shape, W2.shape)
        accuracy = 0
        cost = J(p2, encode_y, W2, N, lam)
        losses.append(cost)

        hidden_loss = np.dot((encode_y - p2), W2.T)
        output_loss = encode_y - p2
        print(X.T.shape, hidden_loss.shape)

        dW1 = 1 / N * X.T.dot(hidden_loss)
        dW2 = 1 / N * np.dot(p1.T, output_loss)
        db1 = 1 / N * np.sum(hidden_loss, axis=0, keepdims=True)
        db2 = 1 / N * np.sum(output_loss, axis=0, keepdims=True)
        print(W1.shape, dW1.shape)

        W1 = W1 - alpha * dW1
        W2 -= alpha * dW2
        b1 -= alpha * db1
        b1 = b1 - alpha * db1
        b2 -= alpha * db2

        for j in range(len(X)):
            if np.argmax(p2[j]) == np.argmax(encode_y[j]):
                accuracy += 1
        accuracy = accuracy / len(encode_y) * 100
        total_acc += accuracy
        accs.append(total_acc)

    plot_data(X, encode_y, W2, b2)

    return losses, total_acc / maxEpochs, accs


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
