import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import torch.optim as optim


SEED = 2147483647
TOTAL_RUNS = 5

DIFF = 10
NUM_EPOCHS = 11
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
    X = X[p]
    Y = Y[p]

    X1 = torch.randn(50, 50)
    X2 = torch.randn(50, 50) + DIFF
    test_X = torch.cat([X1, X2], dim=0)
    Y1 = torch.zeros(50, 1)
    Y2 = torch.ones(50, 1)
    test_Y = torch.cat([Y1, Y2], dim=0)

    p = torch.randperm(100)
    test_X = test_X[p]
    test_Y = test_Y[p]

    return X, Y, test_X, test_Y


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # fc means Fully Connected
        self.fc1 = nn.Linear(50, 500)
        self.fc_new = nn.Linear(500, 400)
        self.fc2 = nn.Linear(400, 340)
        self.out = nn.Linear(340, 1)
        self.out_act = nn.Sigmoid()
        self.hid_act = nn.Tanh()

    # Task 1: Returns the number of trainable parameters used
    def get_param_count(self):
        total_trainable_parameters = 0
        for p in list(self.parameters()):
            edges = 1
            for s in list(p.size()):
                edges = edges * s
            total_trainable_parameters += edges
        return total_trainable_parameters

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.hid_act(a1)

        a_new = self.fc_new(h1)
        h_new = self.hid_act(a_new)

        a2 = self.fc2(h_new)
        h2 = self.hid_act(a2)

        a4 = self.out(h2)
        y = self.out_act(a4)

        return y


def train_epoch(model, opt, criterion, batch_size, inputs, expected):
    model.train()
    losses = []

    opt.zero_grad()
    # (1) Forward
    y_out = model(inputs)
    # (2) Compute diff
    loss = criterion(y_out, expected)
    # (3) Compute gradients
    loss.backward()

    # (4) update weights
    opt.step()

    return loss


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = get_data()

    avg = 0
    for i in range(TOTAL_RUNS):
        # train network
        net = Net()
        crit = nn.BCELoss()

        training_losses = []
        test_losses = []
        adam = optim.Adam(net.parameters(), lr=LR)
        ynp, pred = np.array([1]), np.array([0])
        for e in range(NUM_EPOCHS):
            epoch_loss = train_epoch(model=net, opt=adam, criterion=crit, batch_size=50, inputs=X_train,
                                     expected=Y_train)
            training_losses.append(epoch_loss)

            # Note: this notifies the network that it finished training. We don't actually need this line now
            # since our network is primitive, but it is nice to have good habits for future works
            net.eval()

            # run test set
            out = net(X_test)

            pred = torch.round(out).detach()
            # convert ground truth to numpy
            ynp = Y_test.data

            test_loss = crit(out, ynp)
            test_losses.append(test_loss)

        plt.plot(range(1, NUM_EPOCHS + 1), training_losses, color=[0.99, 0.0, 0.0], label="Train Loss")
        plt.plot(range(1, NUM_EPOCHS + 1), test_losses, color=[0.0, 0.0, 0.99], label="Test Loss")
        plt.legend(loc='upper right')
        plt.show()
        acc = np.count_nonzero(ynp == pred)

        print("(Run {}) -> Model accuracy: {}%".format(i + 1, acc))
        avg += acc

    print("-------------------------------")
    print("Average accuracy -> {}%".format(avg/TOTAL_RUNS))
