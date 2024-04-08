import functools
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)


def df_to_sparse_matrix(df):
    rows = [v - 1 for v in list(df["user"])]
    cols = [v - 1 for v in list(df["item"])]
    # vals = jnp.array([(v - 1) / 4 for v in list(df["rating"])])
    vals = jnp.array(list(df["rating"]))
    return {"indices": (rows, cols), "values": vals}


def train_test_random(train_frac, key):
    df = pd.read_table("../dataset/ml-100k/u.data", names=("user", "item", "rating", "time"))
    key, subkey = jax.random.split(key)
    random_state = jax.random.randint(subkey, (1,), 0, 2**31)[0].item()
    df = df.sample(frac=1, random_state=random_state)
    n_total = len(df)
    n_train = int(n_total * train_frac)
    return df_to_sparse_matrix(df[:n_train]), df_to_sparse_matrix(df[n_train:-1])


def train_test_u1():
    df_train = pd.read_table("../dataset/ml-100k/u1.base", names=("user", "item", "rating", "time"))
    df_test = pd.read_table("../dataset/ml-100k/u1.test", names=("user", "item", "rating", "time"))
    return df_to_sparse_matrix(df_train), df_to_sparse_matrix(df_test)


class Problem:
    gmin_plus_hmin = 0

    def __init__(self, nonneg, train_frac, dim_feature, regularization, sigma_init, seed=0):
        key = jax.random.PRNGKey(seed)
        self.nonneg = nonneg
        self.regularization = regularization

        # self.data_train, self.data_test = train_test_random(train_frac, key)
        self.data_train, self.data_test = train_test_u1()

        self.n_train = len(self.data_train["indices"][0])
        self.n_test = len(self.data_test["indices"][0])
        print(self.n_train, self.n_test)
        num_user = max(self.data_train["indices"][0]) + 1
        num_item = max(self.data_train["indices"][1]) + 1
        # assert df["user"].max() == num_user
        # assert df["item"].max() == num_item

        key, *subkeys = jax.random.split(key, num=3)
        params = [
            jax.random.uniform(subkeys[0], (num_user, dim_feature)) * sigma_init,
            jax.random.uniform(subkeys[1], (num_item, dim_feature)) * sigma_init,
        ]
        self.param_shape = [p.shape for p in params]
        self.param_size_cumsum = np.cumsum([p.size for p in params])
        self.x0 = jnp.concatenate([jnp.float64(p.ravel()) for p in params])
        print("Number of variables:", self.param_size_cumsum[-1])

    def unflatten(self, x):
        return [p.reshape(s) for (s, p) in zip(self.param_shape, jnp.split(x, self.param_size_cumsum))]

    def loss(self, z):
        return jnp.mean(jnp.square(z))

    @functools.partial(jax.jit, static_argnums=(0,))
    def c(self, x):
        u, v = self.unflatten(x)
        rows, cols = self.data_train["indices"]
        z = (u @ v.T)[(rows, cols)] - self.data_train["values"]
        return jnp.concatenate((z, u.reshape(-1), v.reshape(-1)))

    @functools.partial(jax.jit, static_argnums=(0,))
    def h(self, z):
        z0, z1 = jnp.split(z, [self.n_train])
        return self.loss(z0) + self.regularization * (jnp.sum(jnp.square(z1)))

    @functools.partial(jax.jit, static_argnums=(0,))
    def g_prox(self, x, eta):
        if self.nonneg:
            return jnp.maximum(x, 0)
        else:
            return x

    @functools.partial(jax.jit, static_argnums=(0,))
    def g(self, x):
        return 0
