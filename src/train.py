import datetime
from pathlib import Path
from typing import Dict, Tuple

import tensorflow as tf

print("TensorFlow version:", tf.__version__)


class Trainer:
    loss_types = (
        "binarycrossentropy",
        "categoricalcrossentropy",
        "categoricalhinge",
        "cosinesimilarity",
        "hinge",
        "huber",
        "kldivergence",
        "logcosh",
        "loss",
        "meanabsoluteerror",
        "meanabsolutepercentageerror",
        "meansquarederror",
        "meansquaredlogarithmicerror",
        "poisson",
        "reduction",
        "sparsecategoricalcrossentropy",
        "squaredhinge",
    )
    optimiser_types = (
        "adadelta",
        "adagrad",
        "adam",
        "adamax",
        "ftrl",
        "nadam",
        "optimize",
        "rmsprop",
        "sgd",
    )

    def __init__(self,
                 model: tf.keras.Model,
                 model_name: str,
                 train_dataset: Tuple,
                 test_dataset: Tuple,
                 loss_type: str,
                 optimiser_type: str,
                 loss_kwargs: Dict,
                 optimiser_kwargs: Dict,
                 training_out_path: [str, Path] = None):

        self.model: tf.keras.Model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.loss_object: tf.keras.Loss = self._get_loss_object(loss_type, loss_kwargs)
        self.optimiser: tf.keras.optimizers.Optimizer = self._get_optimiser(optimiser_type, optimiser_kwargs)

        if training_out_path:
            self.base_path: Path = Path(training_out_path)
        else:
            self.base_path = Path.cwd().parent / "training" / model_name

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.log_path = self.base_path / "logs"

        self.train_log_path = self.log_path / "gradient_tape" / current_time / "train"
        self.test_log_path = self.log_path / "gradient_tape" / current_time / "test"

        # need to create directories
        [x.mkdir(exist_ok=True, parents=True) for x in [self.train_log_path, self.test_log_path]]

        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_path)
        self.test_summary_writer = tf.summary.create_file_writer(self.test_log_path)

        self.epoch_loss_avg = tf.keras.metrics.Mean()
        self.epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        self.train_loss_results = []
        self.train_accuracy_results = []

    def _get_loss_object(self, loss_type: str, loss_kwargs: Dict) -> tf.keras.Loss:
        if loss_type == "binarycrossentropy":
            loss_object = tf.losses.BinaryCrossentropy(**loss_kwargs)
        elif loss_type == "categoricalcrossentropy":
            loss_object = tf.losses.CategoricalCrossentropy(**loss_kwargs)
        elif loss_type == "categoricalhinge":
            loss_object = tf.losses.CategoricalHinge(**loss_kwargs)
        elif loss_type == "cosinesimilarity":
            loss_object = tf.losses.CosineSimilarity(**loss_kwargs)
        elif loss_type == "hinge":
            loss_object = tf.losses.Hinge(**loss_kwargs)
        elif loss_type == "huber":
            loss_object = tf.losses.Huber(**loss_kwargs)
        elif loss_type == "kldivergence":
            loss_object = tf.losses.KLDivergence(**loss_kwargs)
        elif loss_type == "logcosh":
            loss_object = tf.losses.LogCosh(**loss_kwargs)
        elif loss_type == "loss":
            loss_object = tf.losses.Loss(**loss_kwargs)
        elif loss_type == "meanabsoluteerror":
            loss_object = tf.losses.MeanAbsoluteError(**loss_kwargs)
        elif loss_type == "meanabsolutepercentageerror":
            loss_object = tf.losses.MeanAbsolutePercentageError(**loss_kwargs)
        elif loss_type == "meansquarederror":
            loss_object = tf.losses.MeanSquaredError(**loss_kwargs)
        elif loss_type == "meansquaredlogarithmicerror":
            loss_object = tf.losses.MeanSquaredLogarithmicError(**loss_kwargs)
        elif loss_type == "poisson":
            loss_object = tf.losses.Poisson(**loss_kwargs)
        elif loss_type == "reduction":
            loss_object = tf.losses.Reduction(**loss_kwargs)
        elif loss_type == "sparsecategoricalcrossentropy":
            loss_object = tf.losses.SparseCategoricalCrossentropy(**loss_kwargs)
        elif loss_type == "squaredhinge":
            loss_object = tf.losses.SquaredHinge(**loss_kwargs)
        else:
            raise ValueError(f"Loss type {loss_type}, not a valid loss type. Please try one of {self.loss_types}")

        return loss_object

    def _get_optimiser(self, optimiser_type: str, optimiser_kwargs: Dict) -> tf.keras.optimizers.Optimizer:

        if optimiser_type == "adadelta":
            optimiser = tf.keras.optimizers.Adadelta(**optimiser_kwargs)

        elif optimiser_type == "adagrad":
            optimiser = tf.keras.optimizers.Adagrad(**optimiser_kwargs)

        elif optimiser_type == "adam":
            optimiser = tf.keras.optimizers.Adam(**optimiser_kwargs)

        elif optimiser_type == "adamax":
            optimiser = tf.keras.optimizers.Adamax(**optimiser_kwargs)

        elif optimiser_type == "ftrl":
            optimiser = tf.keras.optimizers.Ftrl(**optimiser_kwargs)

        elif optimiser_type == "nadam":
            optimiser = tf.keras.optimizers.Nadam(**optimiser_kwargs)

        elif optimiser_type == "optimize":
            optimiser = tf.keras.optimizers.Optimize(**optimiser_kwargs)

        elif optimiser_type == "rmsprop":
            optimiser = tf.keras.optimizers.RMSprop(**optimiser_kwargs)

        elif optimiser_type == "sgd":
            optimiser = tf.keras.optimizers.SGD(**optimiser_kwargs)
        else:
            raise ValueError(
                f"Optimiser type {optimiser_type}, not a valid loss type. Please try one of {self.optimiser_types}"
            )

        return optimiser

    def _loss(self, x, y, training=False):
        y_ = self.model(x, training=training)
        return self.loss_object(y_true=y, y_pred=y_)

    def _grad(self, inputs=None, targets=None, training=False) -> Tuple[float, tf.Tensor]:
        with tf.GradientTape() as tape:
            loss_value = self._loss(x=inputs, y=targets, training=training)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def train_step(self, x, y, train_loss_metric, train_acc_metric) -> Tuple[float, tf.Tensor]:
        loss, grads = self._grad(x, y, training=True)
        self.optimiser.apply_gradients(zip(grads, self.model.trainable_variables))
        train_loss_metric.update_state(loss)
        train_acc_metric.update_state(y, self.model(x, training=True))

        return loss, grads

    def test_step(self, x, y, test_loss_metric, test_acc_metric) -> Tuple[float, tf.Tensor]:
        loss, grads = self._grad(x, y, training=False)
        test_loss_metric.update_state(loss)
        test_acc_metric.update_state(y, self.model(x, training=False))

        return loss, grads

    def train(self, n_epochs: int):
        for epoch in range(n_epochs):
            train_epoch_loss_avg = tf.keras.metrics.Mean()
            train_epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            test_epoch_loss_avg = tf.keras.metrics.Mean()
            test_epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

            for x, y in self.train_dataset:
                self.train_step(x, y, train_epoch_loss_avg, train_epoch_accuracy)

                with self.train_summary_writer.as_default(step=epoch):
                    tf.summary.scalar("loss", train_epoch_loss_avg.result())
                    tf.summary.scalar("accuracy", train_epoch_accuracy.result()*100)

            for x, y in self.test_dataset:
                self.test_step(x, y, test_epoch_loss_avg, test_epoch_accuracy)

                with self.test_summary_writer.as_default(step=epoch):
                    tf.summary.scalar("loss", test_epoch_loss_avg.result())
                    tf.summary.scalar("accuracy", test_epoch_accuracy.result()*100)

            train_epoch_loss_avg.reset_states()
            train_epoch_accuracy.reset_states()
            test_epoch_loss_avg.reset_states()
            test_epoch_accuracy.reset_states()
