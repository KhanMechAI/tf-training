
import tensorflow as tf

from trainer import Trainer
from datasets import DatasetLoader
from models import BasicClassifier

def main():

    dataset = DatasetLoader("cifar10")

    n_classes = len(dataset.class_names)

    data_aug_layers = [
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomRotation(factor=(-0.2, 0.3)),
    ]

    data_aug = tf.keras.Sequential(
        data_aug_layers
    )

    dataset.prepare(data_aug)

    classifier = BasicClassifier(n_classes, "Basic BasicClassifier")

    trainer = Trainer(
        model=classifier,
        train_dataset=dataset.train_dataset,
        test_dataset=dataset.test_dataset,
        loss_type="sparsecategoricalcrossentropy",
        optimiser_type="adagrad",
        loss_kwargs=dict(
            from_logits=True,
            reduction=tf.losses.Reduction.AUTO,
            name='sparse_categorical_crossentropy'
        ),
        optimiser_kwargs=dict(
            learning_rate=0.001,
            initial_accumulator_value=0.1,
            epsilon=1e-07,
            name='Adagrad'
        ),
    )

    trainer.train(30)


if __name__ == '__main__':
    main()