import jax.numpy as jnp


class Problem:
    outer_min = 0

    def __init__(self, a, b, d, x0):
        self.a = a
        self.b = b
        self.d = d
        self.x0 = jnp.ones(d) * x0

    def inner_func(self, x):
        return jnp.concatenate((self.a - x[:-1], jnp.sqrt(self.b) * (x[1:] - x[:-1] ** 2))) * jnp.sqrt(2)

    def outer_func(self, r):
        return jnp.linalg.norm(r) ** 2 / 2

    def prox(self, x, eta):
        return x
