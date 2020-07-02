from statsmodels.tsa.arima_process import arma_generate_sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from random import random

MAXIMUM = 200
LSTM_LAYERS = 1
TEST_NUM = 50
scaler = MinMaxScaler(feature_range=(-1, 1))


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
    def __init__(self, input_dim, hidden_dim, target_dim, lr):
        super(LSTMBinaryModel, self).__init__()
        self.loss_function = nn.MSELoss()
        self.hidden_layer_size = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=LSTM_LAYERS)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(self.hidden_layer_size, target_dim)

        self.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, self.hidden_layer_size),
                            torch.zeros(LSTM_LAYERS, 1, self.hidden_layer_size))

        # keep at end
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.training_loss = []
        self.test_loss = []

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

    def gradient_step(self, y_pred, labels):
        loss = self.loss_function(y_pred, labels)
        self.training_loss.append(loss.item())
        loss.backward()
        self.optimizer.step()


def train(model, train_sequence, epochs):
    model.train()

    epoch_timeline = [i for i in range(epochs)]
    epoch_losses = []
    for i in range(epochs):
        y_result = []

        max_index = len(train_sequence[0])
        first, second, label = train_sequence[0], train_sequence[1], train_sequence[2]
        for index in range(max_index):

            model.optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, model.hidden_layer_size),
                                 torch.zeros(LSTM_LAYERS, 1, model.hidden_layer_size))

            seq = torch.tensor([first[index], second[index]])
            y_pred = model(seq)
            y_result.append(y_pred.item())

            model.gradient_step(y_pred, torch.FloatTensor([label[index]]))
        loss_function = nn.MSELoss()
        epoch_losses.append(loss_function(torch.FloatTensor(y_result), torch.FloatTensor(label)).item())

    return epoch_timeline, epoch_losses


def perform_test(model, test_length):
    test_timeline = [i for i in range(TEST_NUM)]
    test_losses = []

    loss_function = nn.MSELoss()

    correction = [0 for i in range(test_length)]
    for i in range(TEST_NUM):
        testData = generate_data(test_length)
        test_input1, test_input2, test_label = testData
        predictions = predict_data(model, [test_input1, test_input2])
        loss_result = loss_function(torch.FloatTensor(test_label), torch.FloatTensor(predictions))
        test_losses.append(loss_result.item())

        for j in range(test_length):
            correction[j] += (predictions[j] == test_label[j])

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
        y_pred = y_pred.tolist()
        y_pred = y_pred[0]
        y_pred = round(y_pred)
        predictions.append(y_pred)
    return predictions


def actual_plot(model):
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Time Step')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    x = [i for i in range(len(model.training_loss))]
    y = model.training_loss
    plt.plot(x, y)
    plt.show()


def plot_test_loss(test_timeline, test_losses):
    plt.title('Test Loss')
    plt.xlabel('Random Binary Generated Sample')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    x = test_timeline
    y = test_losses
    plt.plot(x, y)
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
    plt.show()


def plot_training_loss(epoch_timeline, epoch_losses):
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    loss = nn.MSELoss()
    x = epoch_timeline
    y = epoch_losses
    plt.plot(x, y)
    plt.show()


def main():
    percentage = random()
    if percentage < 0.2:
        percentage = 0.2
    train_length = int((random() + MAXIMUM) * percentage)
    test_length = int(train_length * 0.25)

    trainData = generate_data(train_length)

    model = LSTMBinaryModel(1, 100, 1, 0.03)

    # training
    epoch_timeline, epoch_losses = train(model, trainData, epochs=300)
    plot_training_loss(epoch_timeline, epoch_losses)

    # test
    test_timeline, test_losses, correction = perform_test(model, test_length)
    plot_test_loss(test_timeline, test_losses)
    plot_bit_correct_percentage(correction)


if __name__ == '__main__':
    main()
