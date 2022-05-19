import csv
import math as m
from math import pi


def read_dataset(filename):
    """
    Reads dataset
    :param filename: Name of file
    :return: data
    """
    file = open(filename)
    reader = csv.reader(file)
    header = next(reader)
    rows = []
    for row in reader:
        rows.append(row)
    file.close()
    return rows


def calculate_gauss_dist(data, mu, sigma):
    """
    Calculates Gaussian distribution
    :param data: Data
    :param mu: mean
    :param sigma: standard deviation
    :return:
    """
    exponent = m.exp(-((data - mu) ** 2 / (2 * sigma ** 2)))
    return (1 / (m.sqrt(2 * pi) * sigma)) * exponent


def separate_classes(dataset):
    """
    Splits data into two classes
    :param dataset: Data
    :return: Split of classes
    """
    split = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        label = dataset[i][-1]
        if label not in split:
            split[label] = list()
        split[label].append(vector)
    return split


def mean(numbers):
    """
    Computes average
    :param numbers: Numbers to compute
    :return: Average
    """
    new_n = list()
    for number in numbers:
        if type(number) == str:
            new_n.append(float(number))
    return sum(new_n) / float(len(numbers))


def std(numbers):
    """
    Computes Standard deviation
    :param numbers: Numbers to compute
    :return:  Standard deviation
    """
    new_n = list()
    for number in numbers:
        if type(number) == str:
            new_n.append(float(number))
    avg = mean(new_n)
    variance = sum([(x - avg) ** 2 for x in new_n]) / float(len(numbers) - 1)
    return m.sqrt(variance)


def prior_probability(n, dataset):
    """
    Computes priori probability
    :param n: Number of trues or falses
    :param dataset: Data
    :return: P(T), P(F)
    """
    prob_true = n / len(dataset)
    prob_false = n / len(dataset)
    return prob_true, prob_false


def posteriori_probability(bool_val, split):
    """
    Calculates the posteriori probability
    :param bool_val: True or False value
    :param split: Split of classes
    :return: P(X|True or False)
    """
    sum = 0
    for row in split:
        if row == bool_val:
            sum += 1
    return sum / float(len(split))


def likelihood(data):
    """
    Computes maximum likelihood parameters
    :param data: Dataset
    :return: Parameters computed
    """
    params = []
    for col in range(len(data[0])):
        column = [val[col] for val in data]
        if col == 6 or col == 7:
            params.append((mean(column), std(column), len(column)))
        else:
            params.append((posteriori_probability("True", column),
                           posteriori_probability("False", column)))
    params.pop()
    return params


def train(split):
    """
    Trains the data using naive bayes algorithm
    :param split: Split of classes
    :return: parameters
    """
    params = dict()
    for label, features in split.items():
        params[label] = likelihood(features)
    return params


def calculate_class_probability(params, row, dataset):
    """
    Calculates the probability of each row
    :param params: Parameters computed
    :param row: Each row
    :param dataset: Dataset
    :return: Class probability
    """
    probabilities = {}
    prod = 1
    for key, value in params.items():
        probabilities[key] = prior_probability(params[key][6][2], dataset)
        for i in range(len(value)):
            if i < 6:
                prod *= value[i][0]
                probabilities[key] = prod
            else:
                prod *= calculate_gauss_dist(float(row[i]),
                                             value[i][0], value[i][1])
                probabilities[key] = prod
    return probabilities


def predict(params, row, dataset):
    """
    Predicts output variable
    :param params: Parameters computed
    :param row: row to check
    :param dataset: dataset
    :return: output label
    """
    probabilities = calculate_class_probability(params, row, dataset)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


def calc_accuracy(test_set, predicted):
    """
    Calculates accuracy and error
    :param test_set: Test dataset
    :param predicted: List of predicted values
    :return: Acuuracy and error
    """

    correct = 0
    actual = list()
    for row in test_set:
        actual.append(row[-1])
        for x, y in zip(actual, predicted):
            if x == y:
                correct += 1
    accuracy = correct / (len(test_set))
    error = 100 - accuracy
    return accuracy, error


def main():
    """
    Trains and tests spam filtering dataset using
    Naive bayes classifier
    :return:
    """
    dataset = read_dataset("q3.csv")
    split = separate_classes(dataset)
    params = train(split)
    test_set = read_dataset("q3b.csv")
    predicted = list()
    for row in test_set:
        label = predict(params, row, test_set)
        predicted.append(label)
        print("Original Data:%s, Predicted value :%s" % (row, label), "\n")
    print("Accuracy : ", calc_accuracy(test_set, predicted)[0],
          " Error", calc_accuracy(test_set, predicted)[1])


main()
