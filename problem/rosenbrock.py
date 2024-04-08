import jax
import jax.numpy as jnp
import functools


jax.config.update("jax_enable_x64", True)


class Problem:
    gmin_plus_hmin = 0

    def __init__(self, a, b, d, x0):
        self.a = a
        self.b = b
        self.d = d
        self.x0 = jnp.ones(d) * x0

    @functools.partial(jax.jit, static_argnums=(0,))
    def c(self, x):
        return jnp.concatenate((self.a - x[:-1], jnp.sqrt(self.b) * (x[1:] - x[:-1] ** 2))) * jnp.sqrt(2)

    @functools.partial(jax.jit, static_argnums=(0,))
    def h(self, r):
        return jnp.linalg.norm(r) ** 2 / 2

    @functools.partial(jax.jit, static_argnums=(0,))
    def g(self, x):
        return 0

    @functools.partial(jax.jit, static_argnums=(0,))
    def g_prox(self, x, eta):
        return x
