import jax.numpy as jnp
import scipy.optimize as spopt
import numpy as np
from . import internal


class Base:
    def update_and_check(self, oracle: internal.OracleSub):
        return False


class ScipyOpt(Base):
    def __init__(self, x0, tol):
        self.x = x0
        self.tol = tol

    def update_and_check(self, oracle: internal.OracleSub):
        res = spopt.minimize(oracle.func, self.x, jac=oracle.grad, tol=self.tol)
        self.x = res.x
        return True


# based on Algorithm 31 of [d’Aspremont, et al. (2021)]
class AcceleratedProximalGradient(Base):
    def __init__(self, x0, mu, tol_coef, eta_init, eta_inc, eta_dec):
        self.x = self.z = self.x0 = x0
        self.mu = mu
        self.tol_coef = tol_coef
        self.eta = np.maximum(eta_init, eta_inc * mu)
        self.eta_inc = eta_inc
        self.eta_dec = eta_dec
        self.b = 0
        self.iter = 0
        self.miss_count = 0

    def check(self, oracle: internal.OracleSub, y, grad_y):
        CHECK_INTERVAL = 10
        if self.iter % CHECK_INTERVAL != 0:
            return False
        lhs = np.linalg.norm(oracle.grad(self.x) - grad_y - self.eta * (self.x - y))
        rhs = self.tol_coef * self.mu * np.linalg.norm(self.x - self.x0)
        # print(k, eta, lhs, rhs)
        return lhs <= rhs

    def update_and_check(self, oracle: internal.OracleSub):
        while True:
            b = (1 + 2 * self.eta * self.b + np.sqrt(1 + 4 * self.eta * self.b * (1 + self.mu * self.b))) / (2 * (self.eta - self.mu))
            tau = (b - self.b) * (1 + self.mu * self.b) / (b * (1 + self.mu * self.b) + self.mu * self.b * (b - self.b))
            y = self.x + tau * (self.z - self.x)
            func_y, grad_y = oracle.func_grad(y)
            x = oracle.prox(y - grad_y / self.eta, 1 / self.eta)
            func_x = oracle.func(x)
            u = x - y

            if func_x <= func_y + jnp.vdot(grad_y, u) + self.eta / 2 * jnp.linalg.norm(u) ** 2:
                break
            else:
                self.eta *= self.eta_inc
                self.miss_count += 1

        if jnp.vdot(x - self.x, u) < 0:
            print(f"Bad direction: {self.iter}")
            # k_restart += 1
            # continue

        self.x = x
        self.iter += 1

        if self.check(oracle, y, grad_y):
            return True
        phi = (b - self.b) / (1 + self.mu * b)
        self.z = (1 - self.mu * phi) * self.z + self.mu * phi * y + self.eta * phi * (x - y)
        self.b = b
        self.eta *= self.eta_dec
        return False


# apg_dp2019_alg5
class AcceleratedProximalGradient2(Base):
    def __init__(self, x0, Lf, alpha, iter_num):
        self.x = self.z = self.x0 = x0
        self.Lf = Lf
        self.alpha = alpha
        self.c = 1
        self.theta = 0
        self.iter = 0
        self.iter_num = iter_num
        # p(x) = alpha / 2 * ||x - x_0||^2 + g(x)
        # psi_j(x) = c_j / 2 * ||x - z_j||^2 + theta_j p(x)

    def check(self):
        return self.iter >= self.iter_num

    def update_and_check(self, oracle: internal.OracleSub):
        a = (1 + self.alpha * self.theta + np.sqrt((1 + self.alpha * self.theta) * (1 + (self.alpha + 2 * self.Lf) * self.theta))) / self.Lf
        theta = self.theta + a
        v = oracle.prox(self.z, self.theta / self.c)
        y = (self.theta * self.x + a * v) / theta
        self.x = oracle.prox(y - 1 / self.Lf * oracle.grad(y), 1 / self.Lf)
        self.iter += 1
        if self.check():
            return True
        self.z = (self.c * self.z + a * (self.alpha * self.x0 - oracle.grad(self.x))) / (self.c + a * self.alpha)
        self.c += a * self.alpha
        self.theta = theta
        return False


# # based on Algorithm 1 of https://proceedings.mlr.press/v32/lin14.html
# def apg_lx2014_alg1(func, func_grad, prox, x0, eta_init, eta_inc, eta_dec, mu, termination):
#     x_pre = x = x0
#     alpha_pre = 1
#     eta = np.maximum(eta_init, mu)
#     k = k_missed = k_restart = 0

#     while True:
#         alpha = np.sqrt(mu / eta)
#         y = x + (x - x_pre) * alpha / alpha_pre * (1 - alpha_pre) / (1 + alpha)
#         func_y, grad_y = func_grad(y)
#         z = prox(y - grad_y / eta, eta)
#         func_z = func(z)
#         u = z - y
#         norm_u = np.linalg.norm(u)

#         # backtracking for eta
#         if func_z > func_y + jnp.vdot(grad_y, u) + eta / 2 * norm_u**2:
#             eta *= eta_inc
#             k_missed += 1
#             continue

#         # adaptive restart
#         # see Section 3.2 of [B. O'Donoghue and E. Candes (2015)]
#         if jnp.vdot(z - x, u) < 0:
#             # print(f"Restart at k = {k}")
#             x_pre = x
#             alpha_pre = 1
#             k_restart += 1
#             continue

#         x_pre = x
#         x = z
#         alpha_pre = alpha
#         k += 1
#         if termination(k, x, y, grad_y, eta):
#             break
#         eta *= eta_dec

#     print(f"APG stopped in {k} iterations, missed {k_missed} times, restarted {k_restart} times")
#     return x, eta
