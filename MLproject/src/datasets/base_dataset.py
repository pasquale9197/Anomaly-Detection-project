from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class BaseADDataset(ABC):

    def __init__(self, root: str):
        super().__init__()
        self.root = root

        self.n_classes = 2 #0: normal, 1: outlier
        self.normal_classes = None #tuple with original class that define normal class
        self.outlier_classes = None #tuple with original class that define outlier class

        self.train_set = None #type torch.utils.Esempi.Dataset
        self.test_set = None  # type torch.utils.Esempi.Dataset

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False,
                num_workers: int = 0) -> (DataLoader, DataLoader):
        pass

    def __repr__(self):
        return self.__class__.__name__
