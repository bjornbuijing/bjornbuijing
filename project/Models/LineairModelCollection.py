import gin
from torch import nn


# Define models

class CNN(nn.Module):
    """Example from lessions

    Args:
        nn (_type_): _description_
    """
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
    """Very 'unintelligent' model to see how quickly a model reaches a limit

    Args:
        nn (_type_): _description_
    """
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
    """Test to attempt to quickly overfit a model

    Args:
        nn (_type_): _description_
    """
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
    """Attempts to create a more balanced lineair class

    Args:
        nn (_type_): _description_
    """
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
    """
    Quick test to test gin

    Args:
        testText (str): _description_
    """
    print(testText)
