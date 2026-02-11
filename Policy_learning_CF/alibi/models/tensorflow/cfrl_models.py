"""
This module contains the Tensorflow implementation of models used for the Counterfactual with Reinforcement Learning
experiments for both data modalities (image and tabular).
"""

import tensorflow as tf
import tensorflow.keras as keras
from typing import List

class TimeSeriesClassifier(keras.Model):
    """
    Time series classifier used in the experiments for Counterfactual with Reinforcement Learning.
    The model consists of three 1D convolutional blocks, each followed by BatchNormalization and ReLU activation.
    The convolutional block is followed by a GlobalAveragePooling1D layer, and finally a fully connected layer
    with softmax activation is used to predict the class probabilities.
    """

    def __init__(self, output_dim: int, **kwargs) -> None:
        """
        Constructor.

        Parameters
        ----------
        output_dim
            Output dimension (number of classes).
        """
        super().__init__(**kwargs)

        # First convolutional block
        self.conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding="same")
        self.bn1 = keras.layers.BatchNormalization()
        self.act1 = keras.layers.Activation(activation="relu")

        # Second convolutional block
        self.conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding="same")
        self.bn2 = keras.layers.BatchNormalization()
        self.act2 = keras.layers.Activation("relu")

        # Third convolutional block
        self.conv3 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")
        self.bn3 = keras.layers.BatchNormalization()
        self.act3 = keras.layers.Activation("relu")

        # Global Average Pooling layer
        self.gap_layer = keras.layers.GlobalAveragePooling1D()

        # Output layer
        self.output_layer = keras.layers.Dense(output_dim, activation="softmax")

    def call(self, x: tf.Tensor, training: bool = True, **kwargs) -> tf.Tensor:
        """
        Defines the forward pass of the classifier.

        Parameters
        ----------
        x
            Input tensor (time series data).
        training
            Boolean indicating whether the model is in training mode. Used by BatchNormalization and Dropout layers.

        Returns
        -------
        Output tensor representing class probabilities.
        """
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)

        x = self.gap_layer(x)
        x = self.output_layer(x)
        return x


class TimeEncoderSmall(keras.Model):
    def __init__(self, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        # Balanced regularization - not too strong
        reg = keras.regularizers.l2(1e-4)

        # Balanced architecture
        self.conv1 = keras.layers.Conv1D(
            filters=32, kernel_size=3, activation="relu", padding="same",
            kernel_regularizer=reg
        )
        self.pool1 = keras.layers.MaxPool1D(pool_size=2, padding="same")
        self.dropout1 = keras.layers.Dropout(0.2)

        self.conv2 = keras.layers.Conv1D(
            filters=16, kernel_size=3, activation="relu", padding="same",
            kernel_regularizer=reg
        )
        self.pool2 = keras.layers.MaxPool1D(pool_size=2, padding="same")
        self.dropout2 = keras.layers.Dropout(0.2)

        self.flatten = keras.layers.Flatten()
        # Reasonable intermediate layer
        self.fc_mid = keras.layers.Dense(64, activation="relu",
                                         kernel_regularizer=reg)
        self.dropout3 = keras.layers.Dropout(0.3)
        self.fc_latent = keras.layers.Dense(latent_dim, activation="tanh",
                                            kernel_regularizer=reg)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)

        x = self.flatten(x)
        x = self.fc_mid(x)
        x = self.dropout3(x, training=training)
        x = self.fc_latent(x)
        return x



class TimeDecoderSmall(keras.Model):
    def __init__(self, n_timesteps: int, n_features: int, **kwargs):
        super().__init__(**kwargs)
        self.n_timesteps = n_timesteps
        # Balanced regularization matching encoder
        reg = keras.regularizers.l2(1e-4)

        # Balanced architecture matching encoder
        self.fc = keras.layers.Dense(64, activation="relu",
                                     kernel_regularizer=reg)
        self.dropout1 = keras.layers.Dropout(0.3)

        self.fc2 = keras.layers.Dense((n_timesteps // 4) * 16,
                                      activation="relu",
                                      kernel_regularizer=reg)
        self.reshape = keras.layers.Reshape((n_timesteps // 4, 16))
        self.dropout2 = keras.layers.Dropout(0.2)

        self.conv1 = keras.layers.Conv1D(
            16, 3, padding="same", activation="relu",
            kernel_regularizer=reg
        )
        self.up1 = keras.layers.UpSampling1D(size=2)
        self.dropout3 = keras.layers.Dropout(0.2)

        self.conv2 = keras.layers.Conv1D(
            32, 3, padding="same", activation="relu",
            kernel_regularizer=reg
        )
        self.up2 = keras.layers.UpSampling1D(size=2)

        self.out_conv = keras.layers.Conv1D(
            n_features, 3, padding="same", activation="linear"
        )

    def call(self, x, training=False):
        x = self.fc(x)
        x = self.dropout1(x, training=training)

        x = self.fc2(x)
        x = self.reshape(x)
        x = self.dropout2(x, training=training)

        x = self.conv1(x)
        x = self.up1(x)
        x = self.dropout3(x, training=training)

        x = self.conv2(x)
        x = self.up2(x)
        x = self.out_conv(x)

        # Use TF control flow for dynamic shape handling
        cur_len = tf.shape(x)[1]
        target_len = tf.constant(self.n_timesteps, dtype=cur_len.dtype)

        def _crop():
            return x[:, :self.n_timesteps, :]

        def _pad():
            last_vals = x[:, -1:, :]
            pad_len = target_len - cur_len
            multiples = tf.stack([1, pad_len, 1])
            pad = tf.tile(last_vals, multiples)
            return tf.concat([x, pad], axis=1)

        x = tf.cond(cur_len > target_len,
                    _crop,
                    lambda: tf.cond(cur_len < target_len, _pad, lambda: x))
        return x


class TimeEncoder(keras.Model):
    """
    Time series encoder used in the experiments for the Counterfactual with Reinforcement Learning. The model
    consists of two 1D convolutional layers with ReLU activations, followed by max-pooling layers. Finally,
    a fully connected layer with a tanh nonlinearity is used to map the convoluted features to the latent space.
    The tanh clips the output between [-1, 1], as required in the DDPG algorithm.
    """

    def __init__(self, latent_dim: int, **kwargs) -> None:
        """
        Constructor.

        Parameters
        ----------
        latent_dim
            Latent dimension.
        """
        super().__init__(**kwargs)

        self.conv1 = keras.layers.Conv1D(
            filters=64, kernel_size=3, activation="relu", padding="same"
        )
        self.maxpool1 = keras.layers.MaxPool1D(pool_size=2, padding="same")
        self.conv2 = keras.layers.Conv1D(
            filters=32, kernel_size=3, activation="relu", padding="same"
        )
        self.maxpool2 = keras.layers.MaxPool1D(pool_size=2, padding="same")
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(latent_dim, activation='tanh')

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = self.fc1(self.flatten(x))
        return x


class TimeDecoder(keras.Model):
    def __init__(self, n_timesteps: int, n_features: int, **kwargs):
        super().__init__(**kwargs)
        self.n_timesteps = n_timesteps

        self.fc = keras.layers.Dense(n_timesteps // 4 * 32, activation="relu")
        self.reshape = keras.layers.Reshape((n_timesteps // 4, 32))

        # Use causal padding to avoid future information leakage
        self.conv1 = keras.layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")
        self.conv2 = keras.layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu")
        self.output_layer = keras.layers.Conv1D(n_features, kernel_size=3, padding="causal", activation="linear")

        # Custom upsampling that preserves temporal structure
        self.upsample1 = keras.layers.UpSampling1D(size=2)
        self.upsample2 = keras.layers.UpSampling1D(size=2)

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        x = self.fc(x)
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.upsample1(x)
        x = self.conv2(x)
        x = self.upsample2(x)
        x = self.output_layer(x)

        # Crop to exact length to avoid artifacts from upsampling
        if x.shape[1] > self.n_timesteps:
            x = x[:, :self.n_timesteps, :]
        elif x.shape[1] < self.n_timesteps:
            # Use last value padding instead of zero padding
            last_values = x[:, -1:, :]
            padding_needed = self.n_timesteps - x.shape[1]
            padding = tf.tile(last_values, [1, padding_needed, 1])
            x = tf.concat([x, padding], axis=1)

        return x

class MNISTClassifier(keras.Model):
    """
    MNIST classifier used in the experiments for Counterfactual with Reinforcement Learning. The model consists of two
    convolutional layers having 64 and 32 channels and a kernel size of 2 with ReLU nonlinearities, followed by
    maxpooling of size 2 and dropout of 0.3. The convolutional block is followed by a fully connected layer of 256 with
    ReLU nonlinearity, and finally a fully connected layer is used to predict the class logits (10 in MNIST case).
    """

    def __init__(self, output_dim: int = 10, **kwargs) -> None:
        """
        Constructor.

        Parameters
        ----------
        output_dim
            Output dimension
        """
        super().__init__(**kwargs)

        self.conv1 = keras.layers.Conv2D(64, 2, padding="same", activation="relu")
        self.maxpool1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.dropout1 = keras.layers.Dropout(0.3)
        self.conv2 = keras.layers.Conv2D(32, 2, padding="same", activation="relu")
        self.maxpool2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.dropout2 = keras.layers.Dropout(0.3)
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(256, activation="relu")
        self.fc2 = keras.layers.Dense(output_dim)

    def call(self, x: tf.Tensor, training: bool = True, **kwargs) -> tf.Tensor:
        x = self.dropout1(self.maxpool1(self.conv1(x)), training=training)
        x = self.dropout2(self.maxpool2(self.conv2(x)), training=training)
        x = self.fc2(self.fc1(self.flatten(x)))
        return x


class MNISTEncoder(keras.Model):
    """
    MNIST encoder used in the experiments for the Counterfactual with Reinforcement Learning. The model
    consists of 3 convolutional layers having 16, 8 and 8 channels and a kernel size of 3, with ReLU nonlinearities.
    Each convolutional layer is followed by a maxpooling layer of size 2. Finally, a fully connected layer
    follows the convolutional block with a tanh nonlinearity. The tanh clips the output between [-1, 1], required
    in the DDPG algorithm (e.g., [act_low, act_high]). The embedding dimension used in the paper is 32, although
    this can vary.
    """

    def __init__(self, latent_dim: int, **kwargs) -> None:
        """
        Constructor.

        Parameters
        ----------
        latent_dim
            Latent dimension.
        """
        super().__init__(**kwargs)

        self.conv1 = keras.layers.Conv2D(16, 3, padding="same", activation="relu")
        self.maxpool1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.conv2 = keras.layers.Conv2D(8, 3, padding="same", activation="relu")
        self.maxpool2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.conv3 = keras.layers.Conv2D(8, 3, padding="same", activation="relu")
        self.maxpool3 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(latent_dim, activation='tanh')

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = self.maxpool3(self.conv3(x))
        x = self.fc1(self.flatten(x))
        return x


class MNISTDecoder(keras.Model):
    """
    MNIST decoder used in the Counterfactual with Reinforcement Learning experiments. The model consists of a fully
    connected layer of 128 units with ReLU activation followed by a convolutional block. The convolutional block
    consists fo 4 convolutional layers having 8, 8, 8  and 1 channels and a kernel size of 3. Each convolutional layer,
    except the last one, has ReLU nonlinearities and is followed by an up-sampling layer of size 2. The final layers
    uses a sigmoid activation to clip the output values in [0, 1].
    """

    def __init__(self, **kwargs) -> None:
        """ Constructor. """
        super().__init__(**kwargs)

        self.fc1 = keras.layers.Dense(128, activation="relu")
        self.reshape = keras.layers.Reshape((4, 4, 8))
        self.conv1 = keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu")
        self.up1 = keras.layers.UpSampling2D(size=(2, 2))
        self.conv2 = keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu")
        self.up2 = keras.layers.UpSampling2D(size=(2, 2))
        self.conv3 = keras.layers.Conv2D(8, (3, 3), padding="valid", activation="relu")
        self.up3 = keras.layers.UpSampling2D(size=(2, 2))
        self.conv4 = keras.layers.Conv2D(1, (3, 3), padding="same", activation="sigmoid")

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        x = self.reshape(self.fc1(x))
        x = self.up1(self.conv1(x))
        x = self.up2(self.conv2(x))
        x = self.up3(self.conv3(x))
        x = self.conv4(x)
        return x


class ADULTEncoder(keras.Model):
    """
    ADULT encoder used in the Counterfactual with Reinforcement Learning experiments. The model consists of
    two fully connected layers with ReLU and tanh nonlinearities. The tanh nonlinearity clips the embedding in [-1, 1]
    as required in the DDPG algorithm (e.g., [act_low, act_high]). The layers' dimensions used in the paper are
    128 and 15, although those can vary as they were selected to generalize across many datasets.
    """

    def __init__(self, hidden_dim: int, latent_dim: int, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        hidden_dim
            Hidden dimension.
        latent_dim
            Latent dimension.
        """
        super().__init__(**kwargs)
        self.fc1 = keras.layers.Dense(hidden_dim)
        self.fc2 = keras.layers.Dense(latent_dim)

    def call(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
        x = tf.nn.relu(self.fc1(x))
        x = tf.nn.tanh(self.fc2(x))
        return x


class ADULTDecoder(keras.Model):
    """
    ADULT decoder used in the Counterfactual with Reinforcement Learning experiments. The model consists of
    of a fully connected layer with ReLU nonlinearity, and a multiheaded layer, one for each categorical feature and
    a single head for the rest of numerical features. The hidden dimension used in the paper is 128.
    """

    def __init__(self, hidden_dim: int, output_dims: List[int], **kwargs):
        """
        Constructor.

        Parameters
        ----------
        hidden_dim
            Hidden dimension.
        output_dim
            List of output dimensions.
        """
        super().__init__(**kwargs)

        self.fc1 = keras.layers.Dense(hidden_dim)
        self.fcs = [keras.layers.Dense(dim) for dim in output_dims]

    def call(self, x: tf.Tensor, **kwargs) -> List[tf.Tensor]:
        x = tf.nn.relu(self.fc1(x))
        xs = [fc(x) for fc in self.fcs]
        return xs
