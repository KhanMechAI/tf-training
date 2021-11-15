import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
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
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(n_classes)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)