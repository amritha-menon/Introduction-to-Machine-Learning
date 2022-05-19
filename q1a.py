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


def compute_derivatives(p, y, X, N, lam, W):
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


def gradientCheck(X, W, b, Y, d, k):
    """
    Implements numerical gradient checking
    :param X: Dataset
    :param W: Weight vector
    :param b: bias vector
    :param Y: Target vector
    :param d: number of observed variables
    :param k: number of classes
    :return: new dW and db
    """
    lam = 0.5
    epsilon = 10 ** (-3)
    W = W.flatten()
    b = b.flatten()
    W_len = len(W)
    theta = np.concatenate((W, b))
    dj = []
    for i in range(len(theta)):
        theta[i] += epsilon
        W = theta[:W_len]
        b = theta[W_len:]
        W = np.reshape(W, (d, k))
        b = np.reshape(b, k)
        f = np.add(np.dot(X, W), b)
        p = softmax(f)
        JPlusE = J(p, Y, W, len(X), lam)
        theta[i] = theta[i] - (2 * epsilon)
        JMinusE = J(p, Y, W, len(X), lam)
        dj.append((JPlusE - JMinusE) / (2 * epsilon))
    return np.array(dj[:W_len]).reshape(W.shape), np.array(dj[W_len:]).reshape(b.shape)


def check(X, Y, W, b, d, k, f, N):
    """
    Does gradient checking
    :param X: Dataset
    :param Y: Target vector
    :param W: Weight vector
    :param b: bias vector
    :param d: number of observed variables
    :param k: number of classes
    :param f: class scores
    :param N: number of samples
    :return: condition for gradient checking
    """
    lam = 0.5
    p = softmax(f)
    dj_dw, dj_db = compute_derivatives(p, Y, X, N, lam, W)
    new_W, new_b = gradientCheck(X, W, b, Y, d, k)
    change_in_W = dj_dw - new_W
    change_in_b = dj_db - new_b
    condition = []
    for i in change_in_W:
        for w in i:
            if w < 1e-4:
                condition.append(True)
            else:
                condition.append(False)
    for i in change_in_b:
        if i.any() < 1e-4:
            condition.append(True)
        else:
            condition.append(False)
    for c in condition:
        if not c:
            return


def train(X, y):
    """
    Trains the model using softmax regression
    :param X: Dataset
    :param y: Target vector
    :return: Loss and accuracy
    """
    alpha = 1e-4
    N, d = X.shape

    k = 2
    encode_y = T(y, k)
    W = np.random.random((d, k))
    b = np.random.random((1, k))
    f = np.add(np.dot(X, W), b)
    check(X, encode_y, W, b, d, k, f, N)
    lam = 0.5
    losses = []
    total_acc = 0
    maxEpochs = 1500
    for i in range(maxEpochs):
        accuracy = 0
        f = np.add(np.dot(X, W), b)
        p = softmax(f)
        cost = J(p, encode_y, W, N, lam)
        losses.append(cost)
        dW, db = compute_derivatives(p, encode_y, X, N, lam, W)
        W -= alpha * dW
        b -= alpha * db
        for j in range(len(X)):
            if np.argmax(p[j]) == np.argmax(encode_y[j]):
                accuracy += 1
        accuracy = accuracy / len(encode_y) * 100
        total_acc += accuracy

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


def main():
    """
    Main function: Learns maximum entropy model for xor problem
    :return:
    """
    xor = read_dataset("xor.dat")
    X = xor[["i1", "i2"]].values.astype(np.float32)
    y = pd.factorize(xor['op'])[0]
    errors, accuracy = train(X, y)
    print("Accuracy:", accuracy)
    plt.plot(errors)
    xlabel("Epochs")
    ylabel("Cost")
    plt.show()


main()
