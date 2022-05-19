import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split


class Logistic_Reg_model(torch.nn.Module):
    def __init__(self, no_input_features):
        super(Logistic_Reg_model, self).__init__()
        self.layer1 = torch.nn.Linear(no_input_features, 20)
        self.layer2 = torch.nn.Linear(20, 1)

    def forward(self, x):
        y_predicted = self.layer1(x)
        y_predicted = torch.sigmoid(self.layer2(y_predicted))
        return y_predicted


def binaryClassifier(file):
    y = file['Species'].tolist()
    x = list(zip(file['MFCCs_10'], file['MFCCs_17']))
    n_features = 2  # number of features

    # train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # pre-processing takes place
    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(x_test)
    le = sklearn.preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)

    # building model
    model = Logistic_Reg_model(n_features)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    number_of_epochs = 1000
    for epoch in range(number_of_epochs):
        y_prediction = model(x_train)
        loss = criterion(y_prediction, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % 100 == 0:
            print('epoch:', epoch + 1, ',loss=', loss.item())
    with torch.no_grad():
        # using test set
        y_pred = model(x_test)
        y_pred_class = y_pred.round()
        accuracy = (y_pred_class.eq(y_test).sum()) / float(y_test.shape[0])
        # accuracy
        print("Accuracy is", accuracy.item() * 100)


def main():
    file1 = pd.read_csv("Frogs.csv")
    file2 = pd.read_csv("Frogs-subsample.csv")
    binaryClassifier(file1)
    binaryClassifier(file2)


main()
