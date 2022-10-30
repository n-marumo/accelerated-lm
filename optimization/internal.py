import jax
import jax.numpy as jnp


class Oracle:
    def __init__(self, instance):
        self.c = instance.inner_func
        self.h = instance.outer_func
        self.prox = instance.prox
        self.h_min = instance.outer_min
        self.h_grad = jax.grad(self.h)
        self.h_func_grad = jax.value_and_grad(self.h)
        self.xk = None
        self.c_xk = None
        self.c_jvp_xk = None
        self.c_vjp_xk = None

        self.count_coef = {
            "obj_func": 1,
            "obj_grad": 3,
            "vjp_pre": 2,
            "transpose": 0,
            "jvp": 1,
            "vjp": 1,
        }
        self.count = dict.fromkeys(self.count_coef.keys(), 0)

    @property
    def total_cost(self):
        return sum([self.count_coef[k] * self.count[k] for k in self.count.keys()])

    def __obj_func(self, x):
        return self.h(self.c(x))

    def obj_func(self, x, counted=True):
        if counted:
            self.count["obj_func"] += 1
        return self.__obj_func(x)

    def obj_grad(self, x):
        self.count["obj_grad"] += 1
        return jax.grad(self.__obj_func)(x)

    def update_xk(self, xk):
        self.count["vjp_pre"] += 1
        self.count["transpose"] += 1
        self.xk = xk
        self.c_xk, _c_vjp_xk = jax.vjp(self.c, xk)
        self.c_vjp_xk = lambda v: _c_vjp_xk(v)[0]
        _c_jvp_xk = jax.linear_transpose(self.c_vjp_xk, self.c_xk)
        self.c_jvp_xk = lambda u: _c_jvp_xk(u)[0]

    def gn_func(self, x):
        self.count["jvp"] += 1
        return self.h(self.c_xk + self.c_jvp_xk(x - self.xk))

    def gn_grad(self, x):
        self.count["jvp"] += 1
        self.count["vjp"] += 1
        return self.c_vjp_xk(self.h_grad(self.c_xk + self.c_jvp_xk(x - self.xk)))

    def gn_func_grad(self, x):
        self.count["jvp"] += 1
        self.count["vjp"] += 1
        u = x - self.xk
        h_y, h_grad_y = self.h_func_grad(self.c_xk + self.c_jvp_xk(u))
        return h_y, self.c_vjp_xk(h_grad_y)


class OracleSub:
    def __init__(self, oracle: Oracle, mu):
        self.oracle = oracle
        self.mu = mu
        self.prox = oracle.prox

    def func(self, x):
        u = x - self.oracle.xk
        return self.oracle.gn_func(x) + self.mu / 2 * jnp.linalg.norm(u) ** 2

    def grad(self, x):
        u = x - self.oracle.xk
        return self.oracle.gn_grad(x) + self.mu * u

    def func_grad(self, x):
        u = x - self.oracle.xk
        f, g = self.oracle.gn_func_grad(x)
        return f + self.mu / 2 * jnp.linalg.norm(u) ** 2, g + self.mu * u


# this is slower
class OracleJVP:
    def __init__(self, instance):
        self.c = instance.inner_func
        self.h = instance.outer_func
        self.proj = instance.proj
        self.h_min = instance.outer_min
        self.h_grad = jax.grad(self.h)
        self.h_func_grad = jax.value_and_grad(self.h)
        self.xk = None
        self.mu = None
        self.c_xk = None
        self.c_jvp_xk = None
        self.c_vjp_xk = None

        self.count_coef = {
            "obj_func": 1,
            "obj_grad": 3,
            "linearize": 2,
            "transpose": 1,
            "jvp": 1,
            "vjp": 1,
        }
        self.count = dict.fromkeys(self.count_coef.keys(), 0)

    @property
    def total_cost(self):
        return sum([self.count_coef[k] * self.count[k] for k in self.count.keys()])

    def obj_func(self, x):
        self.count["obj_func"] += 1
        return self.h(self.c(x))

    def obj_grad(self, x):
        self.count["obj_grad"] += 1
        return jax.grad(self.obj_func)(x)

    def update_xk(self, xk):
        self.count["linearize"] += 1
        self.count["transpose"] += 1
        self.xk = xk
        self.c_xk, self.c_jvp_xk = jax.linearize(self.c, xk)
        self.c_vjp_xk = jax.linear_transpose(self.c_jvp_xk, xk)

    def update_mu(self, mu):
        self.mu = mu

    def lm_func(self, x):
        self.count["jvp"] += 1
        u = x - self.xk
        return self.h(self.c_xk + self.c_jvp_xk(u)) + self.mu / 2 * jnp.linalg.norm(u) ** 2

    def lm_grad(self, x):
        self.count["jvp"] += 1
        self.count["vjp"] += 1
        u = x - self.xk
        return self.c_vjp_xk(self.h_grad(self.c_xk + self.c_jvp_xk(u)))[0] + self.mu * u

    def lm_func_grad(self, x):
        self.count["jvp"] += 1
        self.count["vjp"] += 1
        u = x - self.xk
        h_y, h_grad_y = self.h_func_grad(self.c_xk + self.c_jvp_xk(u))
        return h_y + self.mu / 2 * jnp.linalg.norm(u) ** 2, self.c_vjp_xk(h_grad_y)[0] + self.mu * u
