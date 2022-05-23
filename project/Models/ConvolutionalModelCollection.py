from torch import nn
# Define models


class BRBConvolutionalHigh(nn.Module):
    """ Model which quickly overfits, now with convolutional layers

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super().__init__()

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 784),
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

        self.classifier = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.classifier(x)
        logits = self.dense(x)
        return logits


class BRBConvolutionalLayers(nn.Module):
    """ Model which quickly overfits, now with yet more convolutional layers and datapoints
    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super().__init__()

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 1568),
            nn.ReLU(),
            nn.Linear(1568, 600),
            nn.ReLU(),
            nn.Linear(600, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.classifier(x)
        logits = self.dense(x)
        return logits


class BRBConvolutionalLayersDropout(nn.Module):
    """ Model which quickly overfits, attempt to remove overfitting by adding a dropout layer

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super().__init__()

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 1568),
            nn.ReLU(),
            nn.Linear(1568, 600),
            nn.ReLU(),
            nn.Linear(600, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.classifier(x)
        logits = self.dense(x)
        return logits        
