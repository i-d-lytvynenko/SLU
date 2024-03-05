from abc import ABC, abstractmethod

import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from ..types import Directory, Tuple


class BaseExperiment(ABC):
    def __post_init__(self):
        self.artifacts_dir: Directory
        self.preprocessor: BasePreprocessor
        self.artifacts_dir.mkdir(exist_ok=True)

    @abstractmethod
    def train(self) -> None:
        ...

    @abstractmethod
    def plot(self) -> None:
        ...

    def get_data_loaders(self, X: torch.Tensor, y: torch.Tensor, batch_size: int = 32)\
        -> Tuple[data.DataLoader, data.DataLoader]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.preprocessor.fit(X_train)
        X_train = self.preprocessor.transform(X_train)
        X_test = self.preprocessor.transform(X_test)
        self.artifacts_dir.mkdir(exist_ok=True)
        torch.save(self.preprocessor, self.artifacts_dir/'preprocessor.pth')

        train_dataset = data.TensorDataset(X_train, y_train)
        test_dataset = data.TensorDataset(X_test, y_test)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = data.DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, val_loader


class BasePreprocessor(ABC):
    @abstractmethod
    def fit(self, X: torch.Tensor) -> None:
        ...

    @abstractmethod
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        ...


