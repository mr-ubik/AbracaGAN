"""Simple DCGAN for the AbracaGAN talk."""

import os
from typing import Dict

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow.keras as k

tf.enable_eager_execution()

HYPER = {
    "batch_size": 128,
    "epochs": 150,
    "buffer_size": 6000,
    "noise_dims": 100,
    "learning_rate": 0.0002,
    "beta1": 0.5,
}


def logging_summaries(
    summary_writer: tf.contrib.summary.SummaryWriter, logged: Dict
) -> None:
    """
    Define a simple logging procedure to use with AnoGAN stepwise `train()` logging call.

    Args:
        summary_writer: A `tf.contrib.summary.SummaryWriter` previously instatiated.
        logged: `Dict` collections of all the variables logged by the training ops.

    Returns:
        None.

    """
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.image("generated", logged["generated_data"])
        tf.contrib.summary.image("real", logged["real_data"])
        tf.contrib.summary.scalar("generator/loss", logged["gen_loss"])
        tf.contrib.summary.scalar("discriminator/loss", logged["disc_loss"])


def load_fashion_mnist_train_dataset(hyper: Dict) -> tf.data.Dataset:
    (train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], *(28, 28, 1)).astype(
        "float32"
    )

    # We are normalizing the images to the range of [-1, 1]
    train_images = (train_images - 127.5) / 127.5

    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    train_dataset = (
        dataset.shuffle(hyper["buffer_size"])
        .batch(hyper["batch_size"])
        .repeat(hyper["epochs"])
    )
    return train_dataset


def generator_loss(d_gen):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(d_gen), d_gen)


def discriminator_loss(d_real, d_gen):
    # [1,1,...,1] with real output since it is true and we want
    # our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(d_real), logits=d_real
    )

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(d_gen), logits=d_gen
    )

    total_loss = real_loss + generated_loss

    return total_loss


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


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same")
        self.conv2 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1)

    def call(self, x, training=True):
        x = tf.nn.leaky_relu(self.conv1(x))
        x = self.dropout(x, training=training)
        x = tf.nn.leaky_relu(self.conv2(x))
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


class GAN:
    """Base GAN offering a pluggable scaffolding for more complex models."""

    def __init__(
        self, generator: k.models.Model, discriminator: k.models.Model, hyper: Dict
    ) -> None:
        learning_rate = hyper["learning_rate"]
        beta1 = hyper["beta1"]
        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate, beta1)
        self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate, beta1)

        self.generator = generator()
        self.generator.call = tfe.defun(self.generator.call)

        self.discriminator = discriminator()
        self.discriminator.call = tfe.defun(self.discriminator.call)

        model_dir = "./logs"
        self.checkpoint_prefix = os.path.join(model_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator,
        )
        self.summary_writer = tf.contrib.summary.create_file_writer(
            model_dir, flush_millis=10000
        )

    def train(self, dataset, batch_size, noise_dims):
        global_step = tf.train.get_or_create_global_step()
        for step, real_data in enumerate(dataset, start=1):
            noise = tf.random_normal((batch_size, noise_dims))

            # We record all the operations in the tape
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_data = self.generator(noise, training=True)

                d_real = self.discriminator(real_data, training=True)
                d_gen = self.discriminator(generated_data, training=True)

                gen_loss = generator_loss(d_gen)
                disc_loss = discriminator_loss(d_real, d_gen)

            # We retrieve the gradients from our records
            gradients_of_generator = gen_tape.gradient(
                gen_loss, self.generator.variables
            )
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self.discriminator.variables
            )

            # Optimize and apply the gradients
            self.generator_optimizer.apply_gradients(
                zip(
                    gradients_of_generator,
                    self.generator.variables,
                    global_step=global_step,
                )
            )
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.variables)
            )

            LOGGED = {
                "generated_data": generated_data,
                "real_data": real_data,
                "gen_loss": gen_loss,
                "disc_loss": disc_loss,
            }

            if step % 10 == 0:
                print(f"--------------------------")
                print(f"STEP: {step}")
                print(f"D_LOSS: {disc_loss}")
                print(f"G_LOSS: {gen_loss}")
                logging_summaries(self.summary_writer, LOGGED)


def main():
    """
    Executes the training.
    """
    dataset = load_fashion_mnist_train_dataset(HYPER)
    gan = GAN(generator=Generator, discriminator=Discriminator, hyper=HYPER)
    gan.train(dataset, batch_size=HYPER["batch_size"], noise_dims=HYPER["noise_dims"])


if __name__ == "__main__":
    main()
