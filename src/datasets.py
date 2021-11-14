from typing import Tuple

from tensorflow.keras import datasets


class DatasetLoader():
    def __init__(self):
        pass

    def load(self, dataset: str) -> Tuple[Tuple, Tuple]:
        print(f"Loading {dataset} dataset")
        if dataset == "mnist":
            return datasets.mnist.load_data()
        elif dataset == "cifar10":
            return datasets.cifar10.load_data()
        elif dataset == "cifar100":
            return datasets.cifar100.load_data()