import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_data_loader(training=True):
    """
    TODO: implement this function.

    INPUT:
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    shuffle_flag = False

    if training:
        shuffle_flag = True

    dataset = datasets.FashionMNIST(
        "./data", download=True, train=training, transform=transform
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=shuffle_flag
    )
    return data_loader


def build_model():
    """
    TODO: implement this function.

    INPUT:
        None

    RETURNS:
        An untrained neural network model
    """

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT:
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy
        T - number of epochs for training

    RETURNS:
        None
    """

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    total_number_of_batches = len(train_loader)

    for epoch in range(T):
        train_loss = 0.0
        total_values = 0.0
        correct_values = 0.0
        train_accuracy = 0.0

        for data, label in train_loader:
            optimizer.zero_grad()

            y_hat = model(data)
            loss = criterion(y_hat, label)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            y_pred = F.softmax(y_hat, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.tolist()
            label = label.tolist()

            total_values += len(label)
            correct_values += len([item for item in label if item in y_pred])

        del data, label
        del y_hat, loss, y_pred

        train_loss /= total_number_of_batches
        train_accuracy = (correct_values / total_values) * 100

        print(
            "Train Epoch: {} Accuracy: {}/{}({:.2f}%) Loss: {:.3f}".format(
                epoch,
                int(correct_values),
                int(total_values),
                train_accuracy,
                train_loss,
            )
        )

        del train_loss
        del total_values, correct_values
        del train_accuracy


def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
    TODO: implement this function.

    INPUT:
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cross-entropy

    RETURNS:
        None
    """

    model.eval()
    total_number_of_batches = len(test_loader)

    test_loss = 0.0
    total_values = 0.0
    correct_values = 0.0
    test_accuracy = 0.0

    with torch.no_grad():
        for data, label in test_loader:
            y_hat = model(data)

            loss = criterion(y_hat, label)
            test_loss += loss.item()

            y_pred = F.softmax(y_hat, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.tolist()
            label = label.tolist()

            total_values += len(label)
            correct_values += len([item for item in label if item in y_pred])

        del data, label
        del y_hat, loss, y_pred

    test_loss /= total_number_of_batches
    test_accuracy = (correct_values / total_values) * 100

    if show_loss:
        print("Average loss: {:.4f}".format(test_loss))
    print("Accuracy: {:.2f}%".format(test_accuracy))

    del test_loss
    del total_values, correct_values
    del test_accuracy


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT:
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """

    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle Boot",
    ]

    model.eval()
    y_hat = model(test_images)
    y_pred = F.softmax(y_hat, dim=1).squeeze()

    data = y_pred[index]
    confidence_values, index_values = torch.topk(data, k=3)
    confidence_values = confidence_values.tolist()
    index_values = index_values.tolist()

    for confidence, index in zip(confidence_values, index_values):
        print("{}: {:.2f}%".format(class_names[index], confidence * 100))

    del y_hat, y_pred, data
    del confidence_values, index_values
    del confidence, index
