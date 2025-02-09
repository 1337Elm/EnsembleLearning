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
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),

            nn.Linear(hidden_size, output_size),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.network(x)


class ConvolutionalNeuralNetwork(nn.Module):
    """Class for the convolutional neural network, sub-class of nn.Module

        Structure based on leNet-5
    """
    def __init__(self, image_size: int, channels: int):
        """_summary_

        Args:
            image_size (_type_): Image size in pixels,
             expects symmetric pictures. ex. 48x48 --> image_size = 48
            channels (_type_): Channels in input, 1 for grayscale and 3 for RGB
        """
        super(ConvolutionalNeuralNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(channels, image_size,
                    kernel_size=8, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(5,5), stride=2, padding=1),

            nn.Conv2d(image_size, image_size,
                      kernel_size=8, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(5,5), stride=2, padding=1),

            nn.Flatten(1,-1),
            nn.Linear(256, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.network(x)
