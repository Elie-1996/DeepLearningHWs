import torch
import torch.nn as nn
import numpy as np
from math import pow
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as functional

import matplotlib.pyplot as plt

SEED = 2147483647
DIFF = 1
NUM_EPOCHS = 200
LR = 0.001


def torch_len(tensor):
    return list(tensor.size())[0]


def get_data():
    torch.manual_seed(SEED)
    X1 = torch.randn(1000, 50)
    X2 = torch.randn(1000, 50) + DIFF
    X = torch.cat([X1, X2], dim=0)
    Y1 = torch.zeros(1000, 1)
    Y2 = torch.ones(1000, 1)
    Y = torch.cat([Y1, Y2], dim=0)

    p = torch.randperm(2000)
    X=X[p]
    Y=Y[p]

    X1 = torch.randn(50, 50)
    X2 = torch.randn(50, 50) + DIFF
    test_X = torch.cat([X1, X2], dim=0)
    Y1 = torch.zeros(50, 1)
    Y2 = torch.ones(50, 1)
    test_Y = torch.cat([Y1, Y2], dim=0)

    p = torch.randperm(100)
    test_X=test_X[p]
    test_Y=test_Y[p]

    print("X size: ", end="")
    print(X.size())
    print("Y size: ", end="")
    print(Y.size())

    print("X test size: ", end="")
    print(test_X.size())
    print("Y test size: ", end="")
    print(test_Y.size())

    return X, Y, test_X, test_Y


class Net(nn.Module):
    def __init__(self, x, first, second):
        super().__init__()
        # fc means Fully Connected
        self.fc1 = nn.Linear(50, first)
        self.fc_new = nn.Linear(first, x)
        self.fc2 = nn.Linear(x, second)
        self.out = nn.Linear(second, 1)
        self.out_act = nn.Sigmoid()

    # Task 1: Returns the number of parameters used
    def get_param_count(self):
        return self.parameters().__sizeof__()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = functional.relu(a1)

        a_new = self.fc_new(h1)
        h_new = functional.relu(a_new)

        a2 = self.fc2(h_new)
        h2 = functional.relu(a2)

        a4 = self.out(h2)
        y = self.out_act(a4)

        return y


def train_epoch(model, opt, criterion, batch_size, inputs, expected):
    model.train()
    losses = []

    model.zero_grad()
    # (1) Forward
    y_out = model(inputs)
    # (2) Compute diff
    loss = criterion(y_out, expected)
    # (3) Compute gradients
    loss.backward()

    # (4) update weights
    for p in model.parameters():
        p.data -= p.grad.data * LR


if __name__ == '__main__':
    total_runs = 5
    counter = 0
    X_train, Y_train, X_test, Y_test = get_data()
    while True:
        avg = 0
        counter = counter + 1
        x = 400
        first = round(150 + np.random.rand() * 60)
        second = round(70 + np.random.rand() * 45)
        for i in range(total_runs):
            # train network
            net = Net(x, first, second)
            criterion = nn.BCELoss()

            for e in range(NUM_EPOCHS):
                train_epoch(model=net, opt=None, criterion=criterion, batch_size=50, inputs=X_train, expected=Y_train)

            # Note: this notifies the network that it finished training. We don't actually need this line now
            # since our network is primitive, but it is nice to have good habits for future works
            net.eval()

            # run test set
            out = net(X_test)
            pred = torch.round(out).detach().numpy()

            # convert ground truth to numpy
            ynp = Y_test.data.numpy()

            acc = np.count_nonzero(ynp == pred)

            # print("Number of Epochs: {}.".format(NUM_EPOCHS))
            # print("Model accuracy: {}%".format(acc))
            avg += acc

        print(f"Average accuracy for (first, second) = ({first , second}) -> {avg/total_runs}%")
