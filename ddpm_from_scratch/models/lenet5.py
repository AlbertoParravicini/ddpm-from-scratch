import torch.nn as nn
from einops import rearrange

class LeNet5(nn.Module):
    def __init__(self, num_classes: int):
        """
        An implementation of Lenet5. Use ReLU instead of Tanh, and MaxPool2d instead of AveragePool2d.
        Source: https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/

        :param num_classes: number of classes to predict. For example, 10 in MNIST.
        """
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_1 = nn.Linear(256, 120)
        self.relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(120, 84)
        self.relu_2 = nn.ReLU()
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = rearrange(x, "b c h w -> b (c h w)")  # Flatten features for fully connected layers.
        x = self.fc_1(x)
        x = self.relu_1(x)
        x = self.fc_2(x)
        x = self.relu_2(x)
        x = self.fc_3(x)
        return x
