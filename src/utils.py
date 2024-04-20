import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from .types import Union, Optional, Device, Tuple, List


def smooth_data(data: np.ndarray, window_size: int) -> np.ndarray:
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def to_scientific_notation(num: Union[int, float]) -> str:
    str_num = str(num)
    if 'e' in str_num:
        return str_num
    str_num = f'{num:f}'
    int_part, fractional_part = str_num.split('.')
    precision = len(fractional_part.strip('0')) + len(int_part) - 1
    if int_part == '0':
        precision -= 1
    return f'{num:.{precision}e}'


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    model: nn.Module,
    train_loader: data.DataLoader,
    val_loader: data.DataLoader,
    criterion: nn.modules.loss._Loss,
    lr: float = 1e-3,
    n_epochs: int = 50,
    is_verbose: bool = False,
    device: Optional[Device] = None
) -> Tuple[List, List]:
    device = torch.device(device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss_log = []
    val_loss_log = []

    for epoch in range(1, n_epochs + 1):
        start_time = time.time()
        ep_train_loss_log = []
        ep_val_loss_log = []
        for train_batch_i, (train_inputs, train_targets) in enumerate(train_loader):
            train_inputs = train_inputs.to(device)
            train_targets = train_targets.to(device)
            model.train(True)
            optimizer.zero_grad()
            train_outputs = model(train_inputs)

            if isinstance(criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                train_loss = criterion(train_outputs, train_targets.unsqueeze(1))
            else:
                train_loss = criterion(train_outputs, train_targets)

            train_loss.backward()
            optimizer.step()

            if train_batch_i == int(len(train_loader) / 2):
                model.train(False)
                for val_inputs, val_targets in val_loader:
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)
                    val_outputs = model(val_inputs)
                    if isinstance(criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                        val_loss = criterion(val_outputs, val_targets.unsqueeze(1))
                    else:
                        val_loss = criterion(val_outputs, val_targets)

                    ep_val_loss_log.append(val_loss.item())
                    val_loss_log.append(val_loss.item())

            ep_train_loss_log.append(train_loss.item())
            train_loss_log.append(train_loss.item())

        if is_verbose:
            print(f"Epoch {epoch} of {n_epochs} took {time.time() - start_time:.3f}s")
            print(f"\t  learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"\t  training loss: {np.mean(ep_train_loss_log):.4f}")
            print(f"\tvalidation loss: {np.mean(ep_val_loss_log):.4f}")

    return train_loss_log, val_loss_log
