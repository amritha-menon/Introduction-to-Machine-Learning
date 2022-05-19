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


def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector.
    """
    keys = []
    count = 0
    params = []
    num_params = len(parameters) // 2
    for l in range(1, num_params + 1):
        params = params + ["W" + str(l)]
        params = params + ["b" + str(l)]

    for key in params:

        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1, 1))
        keys = keys + [key] * new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta


def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector .
    """

    count = 0
    for key in ["dW1", "db1", "dW2", "db2"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1, 1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta


def gradientCheck(X, Y, gradients, parameters, cost):
    """
    Implements gradient check
    :param X:
    :param Y:
    :param gradients:
    :param parameters:
    :param cost:
    :return:
    """
    epsilon = 1e-4
    parameters_values = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    for i in range(num_parameters):
        thetaPlus = np.copy(parameters_values)
        thetaPlus[i][0] = thetaPlus[i][0] + epsilon
        J_plus[i] = cost

        thetaMinus = np.copy(parameters_values)
        thetaMinus[i][0] = thetaMinus[i][0] + epsilon
        J_minus[i] = cost

        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    return difference


def check(X, Y, W1, W2, b1, b2, d, k, N):
    """
    Does gradient checking
    :param X:
    :param Y:
    :param W1:
    :param W2:
    :param b1:
    :param b2:
    :param d:
    :param k:
    :param N:
    :return:
    """
    lam = 0.5

    f1 = np.add(np.dot(X, W1), b1)
    f2 = np.add(np.dot(X, W2), b2)
    p1 = softmax(f1)
    p2 = softmax(f2)
    cost = J(p2, Y, W2, N, lam)
    parameters = {"W1": W1,
                  "W2": W2,
                  "b1": b1,
                  "b2": b2}
    hidden_loss = np.dot((Y - p2), W2.T)
    output_loss = Y - p2
    dW1 = 1 / N * np.dot(X.T, hidden_loss)
    dW2 = 1 / N * np.dot(p1.T, output_loss)
    db1 = 1 / N * np.sum(hidden_loss, axis=0, keepdims=True)
    db2 = 1 / N * np.sum(output_loss, axis=0, keepdims=True)
    gradients = {"hidden_loss": hidden_loss,
                 "dW1": dW1,
                 "db1": db1,
                 "output_loss": output_loss,
                 "dW2": dW2,
                 "db2": db2}
    difference = gradientCheck(X, Y, gradients, parameters, cost)

    condition = []
    if difference < 1e-4:
        condition.append(False)
    else:
        condition.append(True)
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
    alpha = 0.0001
    N, d = X.shape

    k = 2
    W1 = np.random.random((d, k))
    W2 = np.random.random((d, k))
    b1 = np.random.random((1, k))
    b2 = np.random.random((1, k))
    encode_y = T(y, k)
    check(X, encode_y, W1, W2, b1, b2, d, k, N)
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

        accuracy = 0
        cost = J(p2, encode_y, W2, N, lam)
        losses.append(cost)

        hidden_loss = np.dot((encode_y - p2), W2.T)
        output_loss = encode_y - p2

        dW1 = 1 / N * np.dot(X.T, hidden_loss)
        dW2 = 1 / N * np.dot(p1.T, output_loss)
        db1 = 1 / N * np.sum(hidden_loss, axis=0, keepdims=True)
        db2 = 1 / N * np.sum(output_loss, axis=0, keepdims=True)
        print(W1.shape, dW1.shape)
        W1 -= alpha * dW1
        W2 -= alpha * dW2
        b1 -= alpha * db1
        b2 -= alpha * db2

        for j in range(len(X)):
            if np.argmax(p2[j]) == np.argmax(encode_y[j]):
                accuracy += 1
        accuracy = accuracy / len(encode_y) * 100
        total_acc += accuracy
        accs.append(total_acc)

    return losses, total_acc / maxEpochs, accs


def main():
    """
    Main function: Learns maximum entropy model for xor problem
    :return:
    """
    xor = read_dataset("xor.dat")
    X = xor[["i1", "i2"]].values.astype(np.float32)
    y = pd.factorize(xor['op'])[0]
    errors, accuracy, accs = train(X, y)
    print("Accuracy:", accuracy)
    plt.plot(errors)
    xlabel("Epochs")
    ylabel("Cost")
    plt.show()

    plt.plot(accs)
    xlabel("Epochs")
    ylabel("Accuracy")
    plt.show()


main()
