from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from pathlib import Path


def GetDataSets(batch_size: int, device: str) -> Tuple[DataLoader, DataLoader]:
    '''
    Returns dataloaders for training and test data
    '''    
    data_dir = Path("../../data/raw/")

    training_data = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )
    training_data.data.to(device)
    training_data.targets.to(device)
    test_data = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    test_data.data.to(device)
    test_data.targets.to(device)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def ModuleTest():
    """
    Print a single string to see whether python functions are updated.
    """
    print('Wat doet python als we iets veranderen?!?!?!')

