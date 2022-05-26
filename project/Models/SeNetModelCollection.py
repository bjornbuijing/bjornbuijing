from matplotlib.pyplot import ylabel
from sqlalchemy import true
from torch import nn
import torch


# Define models

class SeREs(nn.Module):
    '''
    Se net implementation, not working as of yet
    '''
    def __init__(self, kernel: tuple, units: int, sqfactor=16):
        super().__init__()
        sq_units = int(units / sqfactor)
        self.Squeeze = nn.AdaptiveAvgPool2d(kernel)
        self.flat = nn.Flatten()
        self.Squeeze = nn.Linear(units, sq_units)
        self.relu = nn.ReLU()
        self.exite = nn.Linear(sq_units, units)
        self.sigmoid = nn.Sigmoid()        

    def forward(self, x: torch.Tensor):
        skip = x
        x = self.Squeeze(x)
        x = self.flat(x)
        x = self.Squeeze(x)
        x = self.relu(x)
        x = self.exite(x)
        x = self.sigmoid(x)
        x = x[..., None, None]
        return x * skip


class SeREs2(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, inputsize: int, rate: int = 16):
        """_summary_

        Args:
            inputsize (int): Size of the image
            rate (int, optional): squeeze/exite ratio . Defaults to 16.
        """
        super().__init__()
        self.Squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(inputsize, inputsize // rate, bias=False),
            nn.ReLU(inplace=true),
            nn.Linear(inputsize // rate, input, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): item which is handles

        Returns:
            _type_: squeezed and exited item
        """
        bs, c, _, _ = x.shape
        out = self.Squeeze(x).view(bs, c)
        out = torch.flatten(out, 1)
        out = self.excite(out).view(bs, c, 1, 1)
        return x * out.expand_as(x)


class BRBSequentialVariable(nn.Module):
    """Attempts to create a more balanced lineair class

    Args:
        nn (_type_): _description_
    """
    def __init__(self) -> None:
        super().__init__()
        self.seNet = SeREs(kernel=784, sqfactor=16)
        self.lin = nn.Sequential(
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(784, 392),
            nn.ReLU(),
            nn.Linear(392, 32), 
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        y = self.seNet.forward(x)
        y = self.lin(y)
        return y