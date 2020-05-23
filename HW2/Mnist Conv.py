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


# examples = enumerate(TEST)
# batch_idx, (example_data, example_targets) = next(examples)
# print("Shape: ", example_data.shape)
#
# if PLOT:
#     fig = plt.figure()
#     for i in range(6):
#         plt.subplot(2, 3, i + 1)
#         plt.tight_layout()
#         plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#         plt.title("Ground Truth: {}".format(example_targets[i]))
#         plt.xticks([])
#         plt.yticks([])
#     # fig


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
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate, momentum=momentum)

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


def train(network, _epoch, train_loader):
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


def run_test(network, test_loader):
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


if __name__ == '__main__':
    low_train, high_train = divide_data_loader(mnist_train, "train")
    low_test, high_test = divide_data_loader(mnist_test, "test")
    base_model_nn = Net(None)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(low_train) for i in range(n_epochs + 1)]

    # train & test
    # m = 1
    # base_model_nn.print_weights(m, "Base Model - Before Training")
    for epoch in range(1, n_epochs + 1):
        train(base_model_nn, epoch, low_train)
        run_test(base_model_nn, low_test)
    # base_model_nn.print_weights(m, "Base Model - After Training")

    # Made Base Model
    secondary_model_nn = Net(base_model_nn)
    # secondary_model_nn.print_weights(m, "Secondary Model - Before Training")
    for epoch in range(1, n_epochs + 1):
        train(secondary_model_nn, epoch, high_train)
        run_test(secondary_model_nn, high_test)
    # secondary_model_nn.print_weights(m, "Secondary Model - After Training")

    # eval
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.plot(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    # fig
