from statsmodels.tsa.arima_process import arma_generate_sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from random import random


def generate_bit():
    bit = 0
    if random() < 0.5:
        bit = 1
    return bit


def generate_binary_data():
    MAXIMUM = 200
    sequence_length = int((random() + MAXIMUM) * random())
    first, second = [], []
    for i in range(sequence_length):
        first.append(generate_bit())
        second.append(generate_bit())

    return first, second


def main():
    pass


if __name__ == '__main__':
    main()

