from __future__ import print_function

from statsmodels.tsa.arima_process import arma_generate_sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


TRAIN_WINDOW = 10
LSTM_LAYERS = 1
scaler = MinMaxScaler(feature_range=(-1, 1))


def create_inout_sequences(input_data):
    inout_seq = []
    L = len(input_data)
    for i in range(L-TRAIN_WINDOW):
        train_seq = input_data[i:i+TRAIN_WINDOW]
        train_label = input_data[i+TRAIN_WINDOW:i+TRAIN_WINDOW+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def generate_normalized_data():
    trainLength = 100
    testLength = 10

    arparams = np.array([0.6, -0.5, -0.2])
    maparams = np.array([])
    noise = 0.1

    # Generate train and test sequences
    ar = np.r_[1, -arparams]  # add zero-lag and negate
    ma = np.r_[1, maparams]  # add zero-lag

    np.random.seed(1000)

    trainData = arma_generate_sample(ar=ar, ma=ma, nsample=trainLength, scale=noise, distrvs=np.random.uniform)
    testData = arma_generate_sample(ar=ar, ma=ma, nsample=testLength, scale=noise, distrvs=np.random.uniform)

    trainDataNormalized = scaler.fit_transform(trainData.reshape(-1, 1))
    trainDataNormalized = torch.FloatTensor(trainDataNormalized).view(-1)
    return trainDataNormalized, testData


def print_shape(t, title=None):
    if title is not None:
        print("{}:\t".format(title), end="")
    if isinstance(t, np.ndarray):
        print(t.shape)
        return

    if isinstance(t, list):
        t = np.asarray(t)
        print(t.shape)
        return
    print(t.detach().numpy().shape)


class LSTMModel(nn.Sequential):
    def __init__(self, input_dim, hidden_dim, target_dim, lr):
        super(LSTMModel, self).__init__()
        self.loss_function = nn.MSELoss()
        self.hidden_layer_size = hidden_dim
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=LSTM_LAYERS)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(self.hidden_layer_size, target_dim)

        self.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, self.hidden_layer_size),
                            torch.zeros(LSTM_LAYERS, 1, self.hidden_layer_size))

        # keep at end
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

    def gradient_step(self, y_pred, labels):
        loss = self.loss_function(y_pred, labels)
        loss.backward()
        self.optimizer.step()


def predict_data(model, test_inputs):
    model.eval()

    for i in range(TRAIN_WINDOW):
        seq = torch.FloatTensor(test_inputs[-TRAIN_WINDOW:])
        with torch.no_grad():
            model.hidden = (torch.zeros(LSTM_LAYERS, 1, model.hidden_layer_size),
                            torch.zeros(LSTM_LAYERS, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())

    actual_predictions = scaler.inverse_transform(np.array(test_inputs[TRAIN_WINDOW:]).reshape(-1, 1))
    return actual_predictions


def train(model, train_inout_seq, epochs):
    model.train()

    for i in range(epochs):
        for seq, labels in train_inout_seq:
            model.optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, model.hidden_layer_size),
                                 torch.zeros(LSTM_LAYERS, 1, model.hidden_layer_size))

            y_pred = model(seq)

            model.gradient_step(y_pred, labels)


def actual_plot(model, test_data, predictions):
    plt.title('Predictions vs Test Loss')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    x = [i for i in range(TRAIN_WINDOW)]
    loss = nn.MSELoss()
    d = predictions.tolist()
    y = [loss(torch.FloatTensor([test_data[i]]), torch.FloatTensor([d[i]])) for i in range(len(test_data))]
    # plt.plot(test_data)
    # plt.plot([i for i in range(TRAIN_WINDOW)], predictions)
    plt.plot(x, y)
    plt.show()


def main():
    trainData, testData = generate_normalized_data()

    labeledTrainingSequence = create_inout_sequences(trainData)

    model = LSTMModel(1, 100, 1, 0.001)
    train(model, labeledTrainingSequence, epochs=300)
    predictions = predict_data(model, trainData[-TRAIN_WINDOW:].tolist())
    actual_plot(model, testData, predictions)

    pass


if __name__ == '__main__':
    main()
