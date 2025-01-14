import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import load_img, img_to_array

img_shape = (480, 720, 3)

def load_dataset(path):
    import glob
    from tqdm import tqdm
    
    input_glob  = sorted(glob.glob(path +'/data/*.png'))
    ground_glob = sorted(glob.glob(path +'/gt/*.png'))

    input_images = []
    ground_truth = []

    for i in tqdm(input_glob):
        img = load_img(i, target_size=img_shape)
        img = img_to_array(img)
        img = img / 255.
        input_images.append(img)

    for j in tqdm(ground_glob):
        img = load_img(j, target_size=img_shape)
        img = img_to_array(img)
        img = img / 255.
        ground_truth.append(img)

    input_images = np.array(input_images)
    ground_truth = np.array(ground_truth)

    return input_images, ground_truth

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# CVAE architecture
latent_dim = 48

encoder_inputs = keras.Input(shape=img_shape)
x = layers.Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(encoder_inputs)
x = layers.Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
x = layers.Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
x = layers.Conv2D(256, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
CVAE_encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
# encoder.summary()


latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(60 * 90 * 256, activation='relu')(latent_inputs)
x = layers.Reshape((60, 90, 256))(x)
x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
decoder_outputs = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
CVAE_decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
# decoder.summary()


class CVAE(keras.Model):
    def __init__(self, encoder= CVAE_encoder, decoder=CVAE_decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, inputs):
        with tf.GradientTape() as tape:

            data, labels = inputs[0]
            
            # Making predictions
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Evaluating Loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(labels, reconstruction), axis=(1, 2)
                )
            )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # kl_weight = 5

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, inputs):
        # Load data
        data, labels = inputs

        # Compute predictions
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        # Compute the reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.mean_squared_error(labels, reconstruction), axis=(1, 2)
            )
        )

        # Compute the KL loss
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        
        total_loss = reconstruction_loss + kl_loss

        # Update the metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        # Return a dictionary mapping metric names to their current value
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    
    def call(self, input, training=False):
        z_mean, z_log_var, z = self.encoder(input)
        reconstruction = self.decoder(z)
        return reconstruction
