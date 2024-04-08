import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as spslin
import scipy.linalg as splin


# min_x  f(x) + g(x),   where f(x) := h(c(x))


class Oracle:
    def __init__(self, instance):
        self.gmin_plus_hmin = instance.gmin_plus_hmin

        self.__g = instance.g
        self.__g_prox = instance.g_prox
        self.__h = instance.h
        self.__c = instance.c

        self.xk = None
        self.c_xk = None
        self.__c_jvp_xk = None
        self.__c_vjp_xk = None
        self.__hess_h_cxk = None

        self.count = dict.fromkeys(
            [
                "g",
                "g_prox",
                "h",
                "h_grad",
                "h_value_and_grad",
                "h_hvp_pre",
                "h_hvp",
                "f",
                "f_grad",
                "f_value_and_grad",
                "c_vjp_pre",
                "transpose",
                "c_jvp",
                "c_vjp",
            ],
            0,
        )

    def reset_count(self):
        self.count = dict.fromkeys(self.count.keys(), 0)

    def g(self, x, counted=True):
        if counted:
            self.count["g"] += 1
        return self.__g(x)

    def g_prox(self, x, eta, counted=True):
        if counted:
            self.count["g_prox"] += 1
        return self.__g_prox(x, eta)

    def h(self, x):
        self.count["h"] += 1
        return self.__h(x)

    def __h_grad(self, x):
        return jax.grad(self.__h)(x)

    def h_grad(self, x):
        self.count["h_grad"] += 1
        return self.__h_grad(x)

    def h_value_and_grad(self, x):
        self.count["h_value_and_grad"] += 1
        return jax.value_and_grad(self.__h)(x)

    def c_jvp_xk(self, u):
        self.count["c_jvp"] += 1
        return self.__c_jvp_xk(u)

    def c_vjp_xk(self, v):
        self.count["c_vjp"] += 1
        return self.__c_vjp_xk(v)

    def __f(self, x):
        return self.__h(self.__c(x))

    def f(self, x, counted=True):
        if counted:
            self.count["f"] += 1
        return self.__f(x)

    def __f_grad(self, x):
        return jax.grad(self.__f)(x)

    def f_grad(self, x, counted=True):
        if counted:
            self.count["f_grad"] += 1
        return self.__f_grad(x)

    def f_value_and_grad(self, x, counted=True):
        if counted:
            self.count["f_value_and_grad"] += 1
        return jax.value_and_grad(self.__f)(x)

    def obj(self, x, counted=True):
        return self.f(x, counted) + self.g(x, counted)

    # update xk for Jacobian-vector products
    def update_xk(self, xk, ABOGN=False):
        self.count["c_vjp_pre"] += 1
        self.count["transpose"] += 1
        self.xk = xk
        self.c_xk, _c_vjp_xk = jax.vjp(self.__c, xk)
        self.__c_vjp_xk = lambda v: _c_vjp_xk(v)[0]
        _c_jvp_xk = jax.linear_transpose(self.__c_vjp_xk, self.c_xk)
        self.__c_jvp_xk = lambda u: _c_jvp_xk(u)[0]
        if ABOGN:
            self.count["h_hvp_pre"] += 1
            _, self.__hess_h_cxk = jax.linearize(self.__h_grad, self.c_xk)

    # oracles for Gauss-Newton model
    def gn(self, x):
        return self.h(self.c_xk + self.c_jvp_xk(x - self.xk))

    def gn_grad(self, x):
        return self.c_vjp_xk(self.h_grad(self.c_xk + self.c_jvp_xk(x - self.xk)))

    def gn_value_and_grad(self, x):
        u = x - self.xk
        h_y, h_grad_y = self.h_value_and_grad(self.c_xk + self.c_jvp_xk(u))
        return h_y, self.c_vjp_xk(h_grad_y)

    # Gauss-Newton Hessian-vector product for ABO
    def gn_hvp_abo(self, v):
        self.count["h_hvp"] += 1
        # return jax.jvp(self.__h_grad, (self.c_xk,), (v,))[1]
        return self.c_vjp_xk(self.__hess_h_cxk(self.c_jvp_xk(v)))


class OracleSub:
    def __init__(self, oracle: Oracle, mu):
        self.oracle = oracle
        self.mu = mu
        self.g = oracle.g
        self.g_prox = oracle.g_prox

    def f(self, x):
        u = x - self.oracle.xk
        return self.oracle.gn(x) + self.mu / 2 * jnp.linalg.norm(u) ** 2

    def f_grad(self, x):
        u = x - self.oracle.xk
        return self.oracle.gn_grad(x) + self.mu * u

    def f_value_and_grad(self, x):
        u = x - self.oracle.xk
        f, f_grad = self.oracle.gn_value_and_grad(x)
        return f + self.mu / 2 * jnp.linalg.norm(u) ** 2, f_grad + self.mu * u


class OracleSubABO2022:
    def __init__(self, oracle: Oracle, B: spslin.LinearOperator, x, f_grad_x, Delta):
        self.oracle = oracle
        self.B = B
        self.x = x
        self.f_grad_x = f_grad_x
        self.Delta = Delta

    def f(self, s):
        return jnp.vdot(self.f_grad_x, s) + jnp.vdot(self.B.dot(s), s) / 2

    def f_grad(self, s):
        return self.f_grad_x + self.B.dot(s)

    def f_value_and_grad(self, s):
        Bs = self.B.dot(s)
        return jnp.vdot(self.f_grad_x, s) + jnp.vdot(Bs, s) / 2, self.f_grad_x + Bs

    def g(self, s):
        return self.oracle.g(self.x + s)

    # based on Section 5.1 of ABO2022 (the last equation on page 917)
    def g_prox(self, q, nu):
        return jnp.clip(
            self.oracle.g_prox(q + self.x, nu) - self.x,
            a_min=-self.Delta,
            a_max=self.Delta,
        )


# for quasi-Newton model
# based on "Compact Representation of BFGS Updating" (p.181)
# of https://doi.org/10.1007/978-0-387-40065-5
class BFGSB:
    def __init__(self, n, memory=5, min_curvature=1e-8):
        self.n = n
        self.m = memory
        self.mc = min_curvature
        self.d = np.zeros((0,))
        self.S = np.zeros((n, 0))
        self.STS = np.zeros((0, 0))
        self.Y = np.zeros((n, 0))
        self.L = np.zeros((0, 0))
        self.A = None
        self.lu_factor = None
        self.empty = True

    def update(self, s: np.array, y: np.array):
        sy = s @ y
        ss = s @ s
        if sy <= self.mc * ss:
            print("skip update")
            return
        self.delta = y @ y / sy
        self.d = np.append(self.d, sy)
        self.L = np.block([[self.L, np.zeros((self.L.shape[0], 1))], [s @ self.Y, 0]])
        STs = self.S.T @ s
        self.STS = np.block([[self.STS, STs.reshape((-1, 1))], [STs.reshape((1, -1)), ss]])
        self.S = np.hstack((self.S, s.reshape((-1, 1))))
        self.Y = np.hstack((self.Y, y.reshape((-1, 1))))

        if len(self.d) > self.m:
            self.d = self.d[1:]
            self.L = self.L[1:, 1:]
            self.S = self.S[:, 1:]
            self.STS = self.STS[1:, 1:]
            self.Y = self.Y[:, 1:]

        # Eq. (7.29)
        self.A = np.hstack((self.delta * self.S, self.Y))
        self.lu_factor = splin.lu_factor(
            np.block(
                [
                    [self.delta * self.STS, self.L],
                    [self.L.T, -np.diag(self.d)],
                ]
            )
        )
        self.empty = False

    def dot(self, v: np.array):
        if self.empty:
            return np.zeros_like(v)
        else:
            return self.delta * v - self.A @ splin.lu_solve(self.lu_factor, self.A.T @ v)


# for Gauss-Newton model
class GaussNewton:
    def __init__(self, oracle: Oracle):
        self.oracle = oracle

    def dot(self, v: np.array):
        return self.oracle.gn_hvp_abo(v)
