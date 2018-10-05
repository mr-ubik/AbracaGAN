"""
Simple DCGAN for the AbracaGAN talk.
"""
from typing import Dict

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow.keras as k

tf.enable_eager_execution()

HYPER = {"batch_size": 128, "epochs": 100, "buffer_size": 6000, "noise_dims": 100}


def load_fashion_mnist_train_dataset(hyper: Dict) -> tf.data.Dataset:
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    # We are normalizing the images to the range of [-1, 1]
    train_images = (train_images - 127.5) / 127.5

    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    train_dataset = (
        dataset.shuffle(hyper["buffer_size"])
        .batch(hyper["batch_size"])
        .repeat(hyper["epochs"])
    )
    return train_dataset


class Generator(tf.keras.Model):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.fc1 = tf.keras.layers.Dense(7 * 7 * 64, use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2DTranspose(
            64, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2DTranspose(
            32, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
        self.batchnorm3 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2DTranspose(
            1, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.fc1(x)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.relu(x)

        x = tf.reshape(x, shape=(-1, 7, 7, 64))

        x = self.conv1(x)
        x = self.batchnorm2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.batchnorm3(x, training=training)
        x = tf.nn.relu(x)

        x = tf.nn.tanh(self.conv3(x))
        return x
