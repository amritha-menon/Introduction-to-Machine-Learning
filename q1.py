import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def scatterPlot(f1, f2, f3, f4, f5, f6, f7, f8):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.scatter(f1, f2, s=80, alpha=0.5, color="pink", label="HylaMinuta")
    plt.scatter(f3, f4, s=80, alpha=0.5, color="purple", label="HypsiboasCinerascens")
    plt.xlabel('MFCCs_10')
    plt.ylabel('MFCCs_17')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(f5, f6, s=80, alpha=0.5, color="pink", label="HylaMinuta")
    plt.scatter(f7, f8, s=80, alpha=0.5, color="purple", label="HypsiboasCinerascens")
    plt.xlabel('MFCCs_10')
    plt.ylabel('MFCCs_17')
    plt.legend()
    plt.show()


def histoGram(f1, f2, f3, f4, f5, f6, f7, f8):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.hist(f1, bins=50, color="pink")
    plt.hist(f2, bins=50, color="purple")
    plt.title("Histogram for HylaMinuta Frogs")

    plt.subplot(2, 2, 2)
    plt.hist(f3, bins=50, color="pink")
    plt.hist(f4, bins=50, color="purple")
    plt.title("Histogram for HypsiboasCinerasc Frogs")

    plt.subplot(2, 2, 3)
    plt.hist(f5, bins=50, color="pink")
    plt.hist(f6, bins=50, color="purple")
    plt.title("Histogram for HylaMinuta Frogs-subsample")

    plt.subplot(2, 2, 4)
    plt.hist(f7, bins=50, color="pink")
    plt.hist(f8, bins=50, color="purple")
    plt.title("Histogram for HypsiboasCinerasc Frogs-subsample")

    plt.show()


def lineGraph(f1, f2, f3, f4, f5, f6, f7, f8):
    f1.sort()
    f2.sort()
    f3.sort()
    f4.sort()
    f5.sort()
    f6.sort()
    f7.sort()
    f8.sort()
    x1 = ["HylaMinuta"] * len(f1)
    x2 = ["HylaMinuta"] * len(f2)
    x3 = ['HypsiboasCinerascens'] * len(f3)
    x4 = ['HypsiboasCinerascens'] * len(f4)
    x5 = ["HylaMinuta"] * len(f5)
    x6 = ["HylaMinuta"] * len(f6)
    x7 = ['HypsiboasCinerascens'] * len(f7)
    x8 = ['HypsiboasCinerascens'] * len(f8)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(x1, f1, color="pink")
    plt.xlabel('Species')
    plt.ylabel('MFCCs_10')
    plt.plot(x2, f2, color="purple")
    plt.xlabel('Species')
    plt.ylabel('MFCCs_17')
    plt.plot(x3, f3, color="pink")
    plt.xlabel('Species')
    plt.ylabel('MFCCs_10')
    plt.plot(x4, f4, color="purple")
    plt.xlabel('Species')
    plt.ylabel('MFCCs_17')

    plt.subplot(2, 1, 2)
    plt.plot(x5, f5, color="pink")
    plt.xlabel('Species')
    plt.ylabel('MFCCs_10')
    plt.plot(x6, f6, color="purple")
    plt.xlabel('Species')
    plt.ylabel('MFCCs_17')
    plt.plot(x7, f7, color="pink")
    plt.xlabel('Species')
    plt.ylabel('MFCCs_10')
    plt.plot(x8, f8, color="purple")
    plt.xlabel('Species')
    plt.ylabel('MFCCs_17')

    plt.show()


def boxPlot(f1, f2, f3, f4, f5, f6, f7, f8):
    plt.figure(figsize=(10, 10))
    plt.subplot(4, 4, 1)
    plt.boxplot(f1)
    plt.subplot(4, 4, 2)
    plt.boxplot(f2)
    plt.subplot(4, 4, 3)
    plt.boxplot(f3)
    plt.subplot(4, 4, 4)
    plt.boxplot(f4)
    plt.subplot(4, 4, 5)
    plt.boxplot(f5)
    plt.subplot(4, 4, 6)
    plt.boxplot(f6)
    plt.subplot(4, 4, 7)
    plt.boxplot(f7)
    plt.subplot(4, 4, 8)
    plt.boxplot(f8)

    plt.show()


def statisticsDisplay(f1, f2, f3, f4):
    # displaying mean
    m1 = np.mean(f1)
    m2 = np.mean(f2)
    m3 = np.mean(f3)
    m4 = np.mean(f4)
    print("The mean of features of first dataset is", m1, m2)
    print("The mean of features of second dataset is", m3, m4)

    # displaying covariance matrix
    cm1 = np.cov(f1)
    cm2 = np.cov(f2)
    cm3 = np.cov(f3)
    cm4 = np.cov(f4)
    print("The covariance matrix of features of first dataset is", cm1, cm2)
    print("The covariance matrix of features of second dataset is", cm3, cm4)

    # displaying standard deviation
    sd1 = np.std(f1)
    sd2 = np.std(f2)
    sd3 = np.std(f3)
    sd4 = np.std(f4)
    print("The standard deviation of features of first dataset is", sd1, sd2)
    print("The standard deviation of features of second dataset is", sd3, sd4)


def main():
    file1 = pd.read_csv("Frogs.csv")
    file2 = pd.read_csv("Frogs-subsample.csv")

    # getting each column
    mfcc10_file1 = file1['MFCCs_10'].tolist()
    mfcc17_file1 = file1['MFCCs_17'].tolist()
    mfcc10_file2 = file2['MFCCs_10'].tolist()
    mfcc17_file2 = file2['MFCCs_17'].tolist()
    species1 = file1['Species'].tolist()
    species2 = file2['Species'].tolist()

    # extracting features
    mfc10HM = list()
    mfc17HM = list()
    mfc10HC = list()
    mfc17HC = list()
    for i in range(len(species1)):
        if species1[i] == "HylaMinuta":
            mfc10HM.append(mfcc10_file1[i])
            mfc17HM.append(mfcc17_file1[i])
        else:
            mfc10HC.append(mfcc10_file1[i])
            mfc17HC.append(mfcc17_file1[i])

    mfc10HM2 = list()
    mfc17HM2 = list()
    mfc10HC2 = list()
    mfc17HC2 = list()
    for i in range(len(species2)):
        if species2[i] == "HylaMinuta":
            mfc10HM2.append(mfcc10_file2[i])
            mfc17HM2.append(mfcc17_file2[i])
        else:
            mfc10HC2.append(mfcc10_file2[i])
            mfc17HC2.append(mfcc17_file2[i])

    scatterPlot(mfc10HM, mfc17HM, mfc10HC, mfc17HC, mfc10HM2, mfc17HM2, mfc10HC2, mfc17HC2)
    histoGram(mfc10HM, mfc17HM, mfc10HC, mfc17HC, mfc10HM2, mfc17HM2, mfc10HC2, mfc17HC2)
    lineGraph(mfc10HM, mfc17HM, mfc10HC, mfc17HC, mfc10HM2, mfc17HM2, mfc10HC2, mfc17HC2)
    boxPlot(mfc10HM, mfc17HM, mfc10HC, mfc17HC, mfc10HM2, mfc17HM2, mfc10HC2, mfc17HC2)
    statisticsDisplay(mfcc10_file1, mfcc17_file1, mfcc10_file2, mfcc17_file2)


if __name__ == "__main__":
    main()
