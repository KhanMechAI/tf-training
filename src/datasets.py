from enum import Enum
from typing import List, Tuple

import keras
from tensorflow.keras import datasets
import tensorflow_datasets as tfds
import tensorflow.keras.layers as kl
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


class InvalidDatasetError(Exception):
    datasets = ("cifar10", "cifar100", "mnitst")

    def __init__(self, dataset, message: str = f"Dataset is not one of {datasets}"):
        self.dataset = dataset
        self.message = message
        super().__init__(self.message)


class DatasetLoader:
    def __init__(self,
                 dataset: str,
                 batch_size=32,
                 shuffle: [int, None] = None,
                 rescale: bool=True
                 ):
        self.dataset_name = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset, self.metadata = self.load(self.dataset_name)
        self.train_dataset = self.dataset["train"]
        self.test_dataset = self.dataset["test"]
        self.rescale()

    @property
    def class_names(self):
        return self.metadata.features["label"].names

    @staticmethod
    def load(dataset: str) -> Tuple[Tuple, Tuple]:
        print(f"Loading {dataset} dataset")

        return tfds.load(
            dataset,
            as_supervised=True,
            with_info=True
        )

    @property
    def image_shape(self):
        return self.train_dataset[0][0].shape

    @property
    def label_shape(self):
        return self.train_dataset[0][1].shape

    def rescale(self):

        rescaler = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1. / 255),
        ])
        self.train_dataset = self.train_dataset.map(
            lambda x, y: (rescaler(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
        self.test_dataset = self.test_dataset.map(
            lambda x, y: (rescaler(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )

    def prepare(self, augmentation: tf.keras.Model = None):

        if self.shuffle:
            self.train_dataset = self.train_dataset.shuffle(self.shuffle)

        if augmentation:
            self.train_dataset = self.train_dataset.map(
                lambda x, y: (augmentation(x, training=True), y),
                num_parallel_calls=AUTOTUNE
            )

        self.train_dataset = self.train_dataset.batch(self.batch_size)
        self.test_dataset = self.test_dataset.batch(self.batch_size)

    def test_batch(self):
        yield next(iter(self.test_dataset))

    def train_batch(self):
        yield next(iter(self.train_dataset))
