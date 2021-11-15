from typing import List

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model

class BasicClassifier(Model):

    def __init__(self, n_classes: int, model_name: [str, None]):
        super(BasicClassifier, self).__init__()
        self.model_name = model_name
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(n_classes)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


class Classifier(Model):

    def __init__(self, n_classes: int, model_name: [str, None]):
        super(Classifier, self).__init__()
        self.model_name = model_name
        self.conv0 = Conv2D(
            filters=16,
            kernel_size=3,
            activation='relu'
        )
        self.max_pool0 = MaxPooling2D()
        self.conv1 = Conv2D(
            filters=32,
            kernel_size=3,
            activation='relu'
        )
        self.max_pool1 = MaxPooling2D()
        self.conv2 = Conv2D(
            filters=64,
            kernel_size=3,
            activation='relu'
        )
        self.max_pool2 = MaxPooling2D()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(n_classes)

    def call(self, x):
        x = self.conv0(x)
        x = self.max_pool0(x)
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
