import os

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import copy
import time

n_epochs = 3
batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 100
PLOT = False

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


mnist_train = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                         transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ]))
mnist_test = torchvision.datasets.MNIST('/files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ]))

TRAIN = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)

TEST = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)



def divide_data_loader(mnist_data, folder):
    low_img, low_label, high_img, high_label = [], [], [], []

    if not os.path.exists(folder):
        for index, (img, label) in enumerate(mnist_data):
            if 0 <= label <= 6:
                low_img.append(img.tolist())
                low_label.append(label)

            else:
                high_img.append(img.tolist())
                high_label.append(label)

            # if index >= 128:
            #     break

        os.mkdir(folder)
        save_files(folder, low_img, low_label, high_img, high_label)
    else:
        low_img, low_label, high_img, high_label = load_files(folder)

    low_tensor_img = torch.Tensor(low_img)
    low_tensor_label = torch.LongTensor(low_label)
    high_tensor_img = torch.Tensor(high_img)
    high_tensor_label = torch.LongTensor(high_label)

    low_dataset = torch.utils.data.TensorDataset(low_tensor_img, low_tensor_label)
    high_dataset = torch.utils.data.TensorDataset(high_tensor_img, high_tensor_label)

    low_loader = torch.utils.data.DataLoader(low_dataset, batch_size=batch_size_train, shuffle=True)
    high_loader = torch.utils.data.DataLoader(high_dataset, batch_size=batch_size_train, shuffle=True)

    return low_loader, high_loader


def load_files(folder):
    low_img = load_file(f"{folder}\\low_img")
    low_label = load_file(f"{folder}\\low_label")
    high_img = load_file(f"{folder}\\high_img")
    high_label = load_file(f"{folder}\\high_label")
    return low_img, low_label, high_img, high_label


def load_file(path):
    return torch.load(f"{path}.pt")


def save_files(folder, low_img, low_label, high_img, high_label):
    save_list(f"{folder}\\low_img", low_img)
    save_list(f"{folder}\\low_label", low_label)
    save_list(f"{folder}\\high_img", high_img)
    save_list(f"{folder}\\high_label", high_label)


def save_list(path, data):
    torch.save(data, f"{path}.pt")


class Net(nn.Module):

    # debugging purposes
    def print_weights(self, m, msg):
        print(f"Printing Learned Weights: {msg}")

        idx = 0
        for param in self.parameters():
            if self.copy_threshold <= idx <= self.copy_threshold + m:
                print("Final Layers:")
                print(param.data)
            elif self.freeze_threshold <= idx <= self.freeze_threshold + m:
                print("Intermediate Layers:")
                print(param.data)
            elif 0 <= idx <= m:
                print("Initial Layers:")
                print(param.data)
            idx += param.numel()
        print("=================================================================")

    def __init__(self, base_model):
        super(Net, self).__init__()

        # neural network structure
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # threshold 1
        self.freeze_threshold = self.parameters_count()
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)

        # threshold 2
        self.copy_threshold = self.parameters_count()
        self.fc2 = nn.Linear(50, 10)

        if base_model:
            self.secondary_constructor(base_model)

        # decide on optimizer
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate,
                                   momentum=momentum)

    def parameters_count(self):
        return sum(p.numel() for p in self.parameters())

    def secondary_constructor(self, base_model):
        # initialize same as base_model
        fc_sd = copy.deepcopy(self.fc2.state_dict())
        self.load_state_dict(base_model.state_dict())
        self.fc2.load_state_dict(fc_sd)

        # freezing layer gradient updates
        self.conv1.weight.requires_grad = False
        self.conv1.bias.requires_grad = False
        self.conv2.weight.requires_grad = False
        self.conv2.bias.requires_grad = False

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train(network, _epoch, train_loader, train_losses, train_counter):
    network.train()
    for _batch_idx, (data, target) in enumerate(train_loader):
        network.optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        network.optimizer.step()
        if _batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                _epoch, _batch_idx * len(data), len(train_loader.dataset),
                100. * _batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (_batch_idx * 64) + ((_epoch - 1) * len(train_loader.dataset)))
        # if you want to save, set a path
        # torch.save(network.state_dict(), '/model.pth')
        # torch.save(optimizer.state_dict(), '/optimizer.pth')


def run_test(network, test_loader, test_losses):
    test_counter = [i * len(test_loader.dataset) for i in range(n_epochs + 1)]

    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_losses, test_counter


def train_specific_model(model, train_data, test_data, msg):
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_data) for i in range(n_epochs + 1)]
    # train & test
    # model.print_weights(1, f"{msg} - Before Training")
    run_test(model, test_data, test_losses)
    for epoch in range(1, n_epochs + 1):
        train(model, epoch, train_data, train_losses, train_counter)
        run_test(model, test_data, test_losses)
    # base_model_nn.print_weights(m, f"{msg} - After Training")
    return train_losses, train_counter, test_losses, test_counter


def plot_graphs(train_losses, train_counter, test_losses, test_counter, title):
    plt.figure(1)
    plt.plot(train_counter, train_losses, color='blue')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.legend(['Train Loss'], loc='upper right')
    plt.title(f"{title} - Train")

    plt.figure(2)
    plt.plot(test_counter, test_losses, color='red')
    plt.xlabel('number of test examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.legend(['Test Loss'], loc='upper right')
    plt.title(f"{title} - Test")

    plt.show()


global_figure_index = 1


def plot_comparison_graphs(graph_data_from_model1, graph_data_from_model2):
    global global_figure_index
    global_figure_index += 2
    m1_train_losses, m1_train_counter, m1_test_losses, m1_test_counter, m1_legend = graph_data_from_model1
    m2_train_losses, m2_train_counter, m2_test_losses, m2_test_counter, m2_legend = graph_data_from_model2

    plt.figure(global_figure_index)
    plt.plot(m1_train_counter, m1_train_losses, color='blue')
    plt.plot(m2_train_counter, m2_train_losses, color='red')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.legend([m1_legend, m2_legend], loc='upper right')
    plt.title("Train Comparison")
    plt.figure(global_figure_index + 1)
    plt.plot(m1_test_counter, m1_test_losses, color='blue')
    plt.plot(m2_test_counter, m2_test_losses, color='red')
    plt.xlabel('number of test examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.legend([m1_legend, m2_legend], loc='upper right')
    plt.title("Test Comparison")

    plt.show()


if __name__ == '__main__':
    low_train, high_train = divide_data_loader(mnist_train, "train")
    low_test, high_test = divide_data_loader(mnist_test, "test")

    base_model_nn = Net(None)
    bm_train_loss, bm_train_counter, bm_test_losses, bm_test_counter = train_specific_model(base_model_nn, low_train, low_test, "Base Model")
    plot_graphs(bm_train_loss, bm_train_counter, bm_test_losses, bm_test_counter, "Base Model")

    secondary_model_nn = Net(base_model_nn)
    sm_start_time = time.time()
    sm_train_loss, sm_train_counter, sm_test_losses, sm_test_counter = train_specific_model(secondary_model_nn, high_train, high_test, "Secondary Model")
    sm_time_passed = time.time() - sm_start_time
    plot_graphs(sm_train_loss, sm_train_counter, sm_test_losses, sm_test_counter, "Secondary Model")

    third_model_nn = Net(None)
    tm_start_time = time.time()
    tm_train_loss, tm_train_counter, tm_test_losses, tm_test_counter = train_specific_model(third_model_nn, high_train, high_test, "Third Model")
    tm_time_passed = time.time() - tm_start_time
    plot_graphs(tm_train_loss, tm_train_counter, tm_test_losses, tm_test_counter, "Third Model")
    plot_comparison_graphs((sm_train_loss, sm_train_counter, sm_test_losses, sm_test_counter, "Secondary"),
                           (tm_train_loss, tm_train_counter, tm_test_losses, tm_test_counter, "Third"))

    print(f"Time it took to train Secondary: {sm_time_passed}")
    print(f"Time it took to train Third: {tm_time_passed}")
