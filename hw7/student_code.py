# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms

import numpy as np


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.first_convolution_layer = nn.Conv2d(
            3, 6, kernel_size=5, stride=1, padding=0
        )
        self.second_convolution_layer = nn.Conv2d(
            6, 16, kernel_size=5, stride=1, padding=0
        )
        self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu_layer = nn.ReLU()
        self.flatten_layer = nn.Flatten()
        self.first_linear_layer = nn.Linear(400, 256)
        self.second_linear_layer = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, self.num_classes)

    def forward(self, x):
        first_convolutional_layer_output = self.pool_layer(
            self.relu_layer(self.first_convolution_layer(x))
        )
        second_convolutional_layer_output = self.pool_layer(
            self.relu_layer(
                self.second_convolution_layer(first_convolutional_layer_output)
            )
        )
        flatten_layer_output = self.flatten_layer(second_convolutional_layer_output)
        first_linear_layer_output = self.relu_layer(
            self.first_linear_layer(flatten_layer_output)
        )
        second_linear_layer_output = self.relu_layer(
            self.second_linear_layer(first_linear_layer_output)
        )
        out = self.classifier(second_linear_layer_output)

        shape_dict = {
            1: list(first_convolutional_layer_output.shape),
            2: list(second_convolutional_layer_output.shape),
            3: list(flatten_layer_output.shape),
            4: list(first_linear_layer_output.shape),
            5: list(second_linear_layer_output.shape),
            6: list(out.shape),
        }

        return out, shape_dict


def count_model_params():
    """
    return the number of trainable parameters of LeNet.
    """
    model = LeNet()
    model_params = 0.0

    for parameter in model.parameters():
        if parameter.requires_grad:
            params = parameter.numel()
            model_params += params
    return model_params / 1e6


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(
        "[Training set] Epoch: {:d}, Average loss: {:.4f}".format(epoch + 1, train_loss)
    )

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print(
        "[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n".format(
            epoch + 1, 100.0 * test_acc
        )
    )

    return test_acc
