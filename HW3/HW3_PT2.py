from statsmodels.tsa.arima_process import arma_generate_sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from random import random

LSTM_LAYERS = 1
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


def generate_data():
    MAXIMUM = 200
    percentage = random()
    if percentage < 0.2:
        percentage = 0.2
    sequence_length = int((random() + MAXIMUM) * percentage)
    training_data = generate_binary_data(sequence_length)
    test_length = int(sequence_length * 0.25)
    test_data = generate_binary_data(test_length)
    return training_data, test_data


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

    for i in range(epochs):
        max_index = len(train_sequence[0]) - 1
        first, second, label = train_sequence[0], train_sequence[1], train_sequence[2]
        for index in range(max_index):

            model.optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, model.hidden_layer_size),
                                 torch.zeros(LSTM_LAYERS, 1, model.hidden_layer_size))

            seq = torch.tensor([first[index], second[index]])
            y_pred = model(seq)

            model.gradient_step(y_pred, torch.FloatTensor([label[index + 1]]))


def predict_data(model, test_inputs):
    first, second = test_inputs[0], test_inputs[1]
    model.eval()

    predictions = []
    max_index = len(first) - 1
    for index in range(max_index):
        seq = torch.tensor([first[index], second[index]])
        y_pred = model(seq)
        y_pred = y_pred.tolist()
        y_pred = y_pred[0]
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


def actual_plot2(test_data, predictions):
    plt.title('Test Loss')
    plt.ylabel('Loss')
    plt.xlabel('Time Step')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    loss = nn.MSELoss()
    x = [i for i in range(len(predictions))]
    y = [loss(torch.FloatTensor([test_data[i]]), torch.FloatTensor([predictions[i]])) for i in range(len(predictions))]
    plt.plot(x, y)
    plt.show()


def main():
    trainData, testData = generate_data()

    model = LSTMBinaryModel(1, 100, 1, 0.03)
    train(model, trainData, epochs=300)

    test_input1, test_input2, test_label = testData
    predictions = predict_data(model, [test_input1, test_input2])
    actual_plot(model)
    actual_plot2(test_label, predictions)
    pass


if __name__ == '__main__':
    main()
