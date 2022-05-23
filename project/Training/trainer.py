import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import gin
from loguru import logger
from typing import Callable, Union, Protocol
from pathlib import Path
from datetime import datetime
import shutil


class GenericModel(Protocol):
    train: Callable
    eval: Callable
    parameters: Callable

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass


def clean_dir(dir: Union[str, Path]) -> None:
    dir = Path(dir)
    if dir.exists():
        logger.info(f"Clean out {dir}")
        shutil.rmtree(dir)
    else:
        dir.mkdir(parents=True)


def write_gin(dir: Path) -> None:
    path = dir / "saved_config.gin"
    with open(path, "w") as file:
        file.write(gin.operative_config_str())        


@gin.configurable
def RunTrainer(
    model,
    train_dataloader: DataLoader, 
    test_dataloader: DataLoader, 
    epochs: int,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    loss_fn: Callable,
    eval_steps: int, 
    device: str,
    log_dir: str = '..\\trained_models\\') -> GenericModel:
    """_summary_
        Returns a trained model and stores the 
    Args:
        model (_type_): _description_
        train_dataloader (DataLoader): _description_
        test_dataloader (DataLoader): _description_
        epochs (int): _description_
        optimizer (torch.optim.Optimizer): _description_
        learning_rate (float): _description_
        loss_fn (Callable): _description_
        eval_steps (int): _description_
        device (str): _description_
        log_dir (str, optional): _description_. Defaults to '..\trained_models\'.

    Returns:
        GenericModel: _description_
    """
    log_dir = Path(log_dir)
    print(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = log_dir / timestamp
    clean_dir(log_dir)
    logger.info(f"Logging to {log_dir}")    
    writer = SummaryWriter(log_dir=log_dir)
    model = model.to(device)
    optimizer_: torch.optim.Optimizer = optimizer(
        model.parameters(), lr=learning_rate
    )  # type: ignore
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for batch in train_dataloader:
            optimizer_.zero_grad()
            input, target = batch
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer_.step()
            train_loss += loss.data.item()
        train_loss /= len(train_dataloader.dataset)
        print(f"Epoch : {epoch} - train loss ={train_loss}")
        writer.add_scalar("Loss/train", train_loss, epoch)

        model.eval()
        test_loss = 0.0
        test_accuracy = 0.0
        for _ in range(eval_steps):
            input, target = next(iter(test_dataloader))
            input = input.to(device)        
            target = target.to(device)
            output = model(input)
            loss = loss_fn(output, target)
            test_loss += loss.data.item()
            test_accuracy += (output.argmax(dim=1) == target).sum()
        datasize = eval_steps * test_dataloader.batch_size
        test_loss /= datasize
        test_accuracy /= datasize
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Loss/accuracy", test_accuracy, epoch)    
        print(f"testloss :{test_loss} -  test accuracy :{test_accuracy}")
        write_gin(log_dir)