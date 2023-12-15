import tensorflow as tf
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

# Loading Cifar-10 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


def build_autoencoder(input_shape):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))

    # Encoder
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))

    # Decoder
    model.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    model.add(layers.Reshape(input_shape))

    return model


def build_vae(input_shape, latent_dim):
    # Encoder
    encoder_inputs = layers.Input(shape=input_shape)
    flat = layers.Flatten()(encoder_inputs)
    x = layers.Dense(256, activation='relu')(flat)

    # Latent space
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    # Sampling
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(
        latent_dim,))([z_mean, z_log_var])

    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(256, activation='relu')(decoder_inputs)
    outputs = layers.Dense(np.prod(input_shape), activation='sigmoid')(x)
    decoded = layers.Reshape(input_shape)(outputs)

    # Model
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z])
    decoder = models.Model(decoder_inputs, decoded)

    vae_outputs = decoder(encoder(encoder_inputs)[2])
    vae = models.Model(encoder_inputs, vae_outputs)

    return vae


input_shape_cifar = x_train.shape[1:]
latent_dim = 64

# Build and compile AE
ae = build_autoencoder(input_shape_cifar)
ae.compile(optimizer='adam', loss='mse')

# Train AE
ae.fit(x_train, x_train, epochs=10, batch_size=128,
       validation_data=(x_test, x_test))

# Build and compile VAE
vae = build_vae(input_shape_cifar, latent_dim)
vae.compile(optimizer='adam', loss='mse')

# Train VAE
vae.fit(x_train, x_train, epochs=10, batch_size=128,
        validation_data=(x_test, x_test))


# Evaluate AE
ae_loss = ae.evaluate(x_test, x_test, verbose=0)
print(f'AE Test Loss: {ae_loss}')

# Evaluate VAE
vae_loss = vae.evaluate(x_test, x_test, verbose=0)
print(f'VAE Test Loss: {vae_loss}')
