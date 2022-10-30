from typing import Sequence

import functools
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST

jax.config.update("jax_enable_x64", True)

N_TARGETS = 10
SCALE_IMAGE = 255


# load data
# from https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
def one_hot(x, k, dtype=np.float64):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=np.float64))


# MLP
# from https://github.com/google/flax
class MultiLayerPerceptron_sigmoid(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.sigmoid(nn.Dense(feat, dtype=jnp.float64)(x))
        x = nn.Dense(self.features[-1], dtype=jnp.float64)(x)
        return x


class MultiLayerPerceptron_relu(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat, dtype=jnp.float64)(x))
        x = nn.Dense(self.features[-1], dtype=jnp.float64)(x)
        return x


def get_data(train=True, train_size=0, random_state=0):
    data = MNIST("/tmp/mnist/", train=train, download=True, transform=FlattenAndCast())
    images = jnp.array(data.train_data).reshape(len(data.train_data), -1) / SCALE_IMAGE
    labels = one_hot(np.array(data.train_labels), N_TARGETS)
    if train:
        images, _, labels, _ = train_test_split(images, labels, stratify=labels, train_size=train_size, random_state=random_state)
    return jnp.array(images), jnp.array(labels)


class Problem:
    outer_min = 0

    def __init__(self, activation, layer_size, train_size, seed=0):
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        random_state = jax.random.randint(subkey, (1,), 0, 2**31)[0].item()
        self.images_train, self.labels_train = get_data(train=True, train_size=train_size, random_state=random_state)
        self.images_test, self.labels_test = get_data(train=False)
        print(self.images_train.shape)
        print(self.labels_train.shape)
        print(self.images_test.shape)
        print(self.labels_test.shape)

        # build model
        if activation == "sigmoid":
            self.model = MultiLayerPerceptron_sigmoid(layer_size + [N_TARGETS])
        elif activation == "relu":
            self.model = MultiLayerPerceptron_relu(layer_size + [N_TARGETS])
        key, subkey = jax.random.split(key)
        tree = self.model.init(key, self.images_train)

        # flatten parameter
        params, self.treedef = jax.tree_util.tree_flatten(tree)
        self.param_shape = [p.shape for p in params]
        self.param_size_cumsum = np.cumsum([p.size for p in params])
        self.x0 = jnp.concatenate([jnp.float64(p.ravel()) for p in params])
        print("Number of variables:", self.param_size_cumsum[-1])

    def unflatten(self, x):
        return jax.tree_util.tree_unflatten(self.treedef, [p.reshape(s) for (s, p) in zip(self.param_shape, jnp.split(x, self.param_size_cumsum))])

    def loss(self, z, labels):
        return -jnp.vdot(nn.log_softmax(z, axis=1), labels) / labels.shape[0]

    def neural_net(self, x, images):
        return self.model.apply(self.unflatten(x), images)

    @functools.partial(jax.jit, static_argnums=(0,))
    def inner_func(self, x):
        return self.neural_net(x, self.images_train)

    @functools.partial(jax.jit, static_argnums=(0,))
    def outer_func(self, z):
        return self.loss(z, self.labels_train)

    @functools.partial(jax.jit, static_argnums=(0,))
    def prox(self, x, eta):
        return x

    @functools.partial(jax.jit, static_argnums=(0,))
    def test(self, x):
        output = self.neural_net(x, self.images_test)
        predicted_class = jnp.argmax(output, axis=1)
        target_class = jnp.argmax(self.labels_test, axis=1)
        return {
            "test_loss": self.loss(output, self.labels_test),
            "test_accuracy": jnp.mean(predicted_class == target_class),
        }
