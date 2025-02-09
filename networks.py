import torch.nn as nn


class NeuralNetwork(nn.Module):
    """Class for the neural network, sub-class of nn.Module
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """_summary_

        Args:
            input_size (int): Number of inputs to model,
             corresponds to number of features
            hidden_size (int): Number of hidden connections between each layer
            output_size (int): Number of outputs
        """
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)

        return x


class ConvolutionalNeuralNetwork(nn.Module):
    """Class for the convolutional neural network, sub-class of nn.Module

        Structure based on leNet 5
    """
    def __init__(self, image_size: int, channels: int):
        """_summary_

        Args:
            image_size (_type_): Image size in pixels,
             expects symmetric pictures. ex. 48x48 --> image_size = 48
            channels (_type_): Channels in input, 1 for grayscale and 3 for RGB
        """
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(channels, image_size,
                               kernel_size=5, stride=1, padding=1)
        self.activate1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(image_size, image_size,
                               kernel_size=5, stride=1, padding=1)
        self.activate2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten(1, -1)
        self.layer3 = nn.Linear(12544, 1028)
        self.activate3 = nn.ReLU()
        self.layer4 = nn.Linear(1028, 64)
        self.activate4 = nn.ReLU()
        self.layer5 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.activate1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.activate2(x)
        x = self.flat(x)
        x = self.layer3(x)
        x = self.activate3(x)
        x = self.layer4(x)
        x = self.activate4(x)
        x = self.layer5(x)

        return x
