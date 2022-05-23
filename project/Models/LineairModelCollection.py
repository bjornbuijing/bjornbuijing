import gin
from torch import nn


# Define models

class CNN(nn.Module):
    '''
    Convolutional network class
    '''
    def __init__(self):
        super().__init__()

        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.convolutions(x)
        logits = self.dense(x)
        return logits


class BRBSequentialLow(nn.Module):
    '''
    Just lineair network class
    '''
    def __init__(self):
        super().__init__()

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 392),
            nn.ReLU(),
            nn.Linear(392, 10)
        )

    def forward(self, x):
        logits = self.dense(x)
        return logits


class BRBSequentialHigh(nn.Module):
    '''
    Just lineair network class
    '''
    def __init__(self):
        super().__init__()

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 784),
            nn.ReLU(),
            nn.Linear(784, 784),
            nn.ReLU(),
            nn.Linear(784, 784),
            nn.ReLU(),
            nn.Linear(784, 784),
            nn.ReLU(),
            nn.Linear(784, 784),
            nn.ReLU(),
            nn.Linear(784, 784),                        
            nn.ReLU(),
            nn.Linear(784, 392),
            nn.ReLU(),
            nn.Linear(392, 10)
        )

    def forward(self, x):
        logits = self.dense(x)
        return logits

class BRBSequentialVariable(nn.Module):
    '''
    Just lineair network class
    '''
    def __init__(self):
        super().__init__()

        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 392),
            nn.ReLU(),
            nn.Linear(392, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        logits = self.lin(x)
        return logits        


@gin.configurable
def TestGin(testText: str):
    print(testText)


