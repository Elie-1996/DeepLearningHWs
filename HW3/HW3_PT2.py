from statsmodels.tsa.arima_process import arma_generate_sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from random import random
from statistics import mean

TEST_LENGTH_PERCENTAGE = 0.05

MINIMUM = 50
MAXIMUM = 70
TEST_NUM = 20

TRAIN_SIZE = 100
LSTM_LAYERS = 1
HIDDEN_DIM_G = 16
DROPOUT_RATE = 0.0
LEARNING_RATE = 0.006
TOTAL_EPOCS = 40

scaler = MinMaxScaler(feature_range=(-1, 1))
the_loss_function = nn.MSELoss()


def generate_bit():
    bit = "0"
    if random() < 0.5:
        bit = "1"
    return bit


def generate_binary_data(sequence_length):
    first, second = "", ""
    for i in range(sequence_length):
        first += (generate_bit())
        second += (generate_bit())

    label = bin(int(first, 2) + int(second, 2))
    label = label[2:]

    label = [0. for i in range(sequence_length - len(label) + 1)] + [float(char) for char in label]
    first = [0.] + [float(char) for char in first]
    second = [0.] + [float(char) for char in second]
    first.reverse()
    second.reverse()
    label.reverse()
    return [first, second, label]


def generate_data(length):
    return generate_binary_data(length)


class LSTMBinaryModel(nn.Sequential):

    def __init__(self, input_dim, hidden_dim, output_size, drop_prob, lr):
        super(LSTMBinaryModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, self.hidden_dim),
                            torch.zeros(LSTM_LAYERS, 1, self.hidden_dim))

        self.input_dim = input_dim
        self.output_size = output_size
        self.n_layers = LSTM_LAYERS

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, dropout=drop_prob)
        # self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.training_loss = []
        self.test_loss = []

    def forward(self, x):
      x = x.view(1, 1, len(x))
      lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)

      (ht, ct) = self.hidden_cell
      out = self.sigmoid(self.fc(ht[-1]))

      out = out[-1, :]
      return out

    def gradient_step(self, y_pred, labels):
        loss_function = the_loss_function
        loss = loss_function(y_pred, labels)
        loss.backward()
        self.optimizer.step()
        self.training_loss.append(loss.item())


def generate_label(label):
    if label == 0:
        return torch.FloatTensor([1.0, 0.0])
    else:
        return torch.FloatTensor([0.0, 1.0])


def train(model, train_length, epochs):
    model.train()

    epoch_timeline = [i for i in range(epochs)]
    epoch_losses = []
    full_train_sequence = [generate_data(train_length) for i in range(TRAIN_SIZE)]
    for i in range(epochs):
        if i % 10 == 0:
            print(f"Epoch: {i}")
        current_sequence_loss = []
        for train_sequence in full_train_sequence:
            y_result = []
            max_index = len(train_sequence[0])
            first, second, label = train_sequence[0], train_sequence[1], train_sequence[2]
            for index in range(max_index):
                model.optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, model.hidden_dim),
                                     torch.zeros(LSTM_LAYERS, 1, model.hidden_dim))

                seq = torch.tensor([first[index], second[index]])
                y_pred = model(seq)
                _, pred = y_pred.max(0)
                y_result.append(pred.item())

                label_expectation = generate_label(label[index])
                model.gradient_step(y_pred, label_expectation)
            loss_function = the_loss_function
            current_sequence_loss.append(loss_function(torch.FloatTensor(y_result), torch.FloatTensor(label)).item())

        epoch_losses.append(mean(current_sequence_loss))

    return epoch_timeline, epoch_losses


def perform_test(model, test_length):
    test_timeline = [i for i in range(TEST_NUM)]
    test_losses = []

    loss_function = the_loss_function

    correction = [0.0 for _ in range(test_length)]
    for i in range(TEST_NUM):
        testData = generate_data(test_length)
        test_input1, test_input2, test_label = testData

        predictions = predict_data(model, [test_input1, test_input2])
        a = torch.FloatTensor(test_label)
        b = torch.FloatTensor(predictions)
        loss_result = loss_function(a, b)
        test_losses.append(loss_result.item())

        for j in range(test_length):
            correction[j] += float((predictions[j] == test_label[j]))

    for m in range(test_length):
        correction[m] = correction[m] / TEST_NUM

    return test_timeline, test_losses, correction


def predict_data(model, test_inputs):
    first, second = test_inputs[0], test_inputs[1]
    model.eval()

    predictions = []
    max_index = len(first)
    for index in range(max_index):
        seq = torch.tensor([first[index], second[index]])
        y_pred = model(seq)
        _, pred = y_pred.max(0)
        predictions.append(pred.item())
    return predictions


def plot_test_loss(test_timeline, test_losses):
    plt.title('Test Loss')
    plt.xlabel('Random Binary Generated Sample')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    x = test_timeline
    y = test_losses
    plt.plot(x, y)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.show()


def plot_bit_correct_percentage(correction):
    plt.title('Bit Prediction Accuracy')
    plt.xlabel('Bit Index')
    plt.ylabel('Bit Accuracy')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    x = [i for i in range(len(correction))]
    y = correction
    plt.plot(x, y)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.show()


def plot_training_loss(epoch_timeline, epoch_losses):
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    loss = the_loss_function
    x = epoch_timeline
    y = epoch_losses
    plt.plot(x, y)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.show()


def main():
    percentage = random()
    if percentage < 0.2:
        percentage = 0.2
    train_length = int(MINIMUM + ((random() + (MAXIMUM - MINIMUM)) * percentage))
    test_length = int(train_length * TEST_LENGTH_PERCENTAGE)
    test_length = 8
    print("trainlength = " + str(train_length))
    print("testlength = " + str(test_length))

    model = LSTMBinaryModel(2, HIDDEN_DIM_G, 2, DROPOUT_RATE, LEARNING_RATE)

    # training
    epoch_timeline, epoch_losses = train(model, train_length, epochs=TOTAL_EPOCS)
    plot_training_loss(epoch_timeline, epoch_losses)

    # test
    test_timeline, test_losses, correction = perform_test(model, test_length)
    plot_test_loss(test_timeline, test_losses)
    plot_bit_correct_percentage(correction)


if __name__ == '__main__':
    main()
