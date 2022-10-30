import jax.numpy as jnp
import numpy as np
from . import internal, submethod

TOL_INNER_EXACT = 1e-10


class Base:
    def __init__(self):
        self.iter = 0
        self.x = None
        self.func_x = None
        self.optimality = None

    def initialize(self, oracle: internal.Oracle, x0):
        self.x = x0
        self.func_x = oracle.obj_func(x0)

    @property
    def params(self):
        return {}

    def update(self, oracle: internal.Oracle):
        pass


class ProximalGradient(Base):
    def __init__(self, eta_init=1e-3, eta_inc=2, eta_dec=0.95):
        super().__init__()
        self.eta = eta_init
        self.eta_inc = eta_inc
        self.eta_dec = eta_dec

    @property
    def params(self):
        return {"eta": self.eta}

    def update(self, oracle: internal.Oracle):
        grad_x = oracle.obj_grad(self.x)
        while True:
            y = oracle.prox(self.x - grad_x / self.eta, 1 / self.eta)
            func_y = oracle.obj_func(y)
            u = y - self.x
            norm_u = np.linalg.norm(u)
            if func_y <= self.func_x + jnp.vdot(grad_x, u) + self.eta / 2 * norm_u**2:
                break
            else:
                self.eta *= self.eta_inc
        self.optimality = self.eta * norm_u
        self.x = y
        self.func_x = func_y
        self.eta *= self.eta_dec
        self.iter += 1


class DL2018(Base):
    def __init__(self, t=1e3, q=0.5, sigma=1):
        super().__init__()
        self.t = t
        self.q = q
        self.sigma = sigma

    @property
    def params(self):
        return {"mu": 1 / self.t}

    def initialize(self, oracle: internal.Oracle, x0):
        super().initialize(oracle, x0)
        self.xk = x0
        self.func_xk = self.func_x
        oracle.update_xk(self.xk)
        self.set_subproblem(oracle)

    def set_subproblem(self, oracle: internal.Oracle):
        self.oracle_sub = internal.OracleSub(oracle, 1 / self.t)
        self.alg_sub = submethod.ScipyOpt(
            self.x,
            TOL_INNER_EXACT,
        )

    def update(self, oracle: internal.Oracle):
        self.alg_sub.update_and_check(self.oracle_sub)
        self.x = self.alg_sub.x
        f = oracle.obj_func(self.x)
        norm_u = np.linalg.norm(self.x - self.xk)
        if f <= self.func_xk - self.sigma / 2 * (norm_u / self.t) ** 2:
            self.optimality = norm_u / self.t
            self.xk = self.x
            self.func_xk = self.func_x = f
            oracle.update_xk(self.xk)
            self.iter += 1
        else:
            self.t *= self.q
            self.x = self.xk
            self.func_x = self.func_xk
        self.set_subproblem(oracle)


class DP2019(Base):
    def __init__(self, t=1e3, lip_sub=1e0):
        super().__init__()
        self.t = t
        self.lip_sub = lip_sub
        self.iter_num_sub = 1 + np.sqrt(t * lip_sub / 2) * np.log(t * lip_sub)
        self.iter_sub = 0
        self.iter_sub_total = 0

    @property
    def params(self):
        return {
            "iter_sub_total": self.iter_sub_total,
            "mu": 1 / self.t,
        }

    def initialize(self, oracle: internal.Oracle, x0):
        super().initialize(oracle, x0)
        self.xk = x0
        self.func_xk = self.func_x
        oracle.update_xk(self.xk)
        self.set_subproblem(oracle)

    def set_subproblem(self, oracle: internal.Oracle):
        self.oracle_sub = internal.OracleSub(oracle, 1 / self.t)
        self.alg_sub = submethod.AcceleratedProximalGradient2(
            self.x,
            self.lip_sub,
            1 / self.t,
            self.iter_num_sub,
        )

    def update(self, oracle: internal.Oracle):
        terminated_sub = self.alg_sub.update_and_check(self.oracle_sub)
        self.x = self.alg_sub.x
        if terminated_sub:
            norm_u = np.linalg.norm(self.x - self.xk)
            self.optimality = norm_u / self.t
            self.xk = self.x
            self.func_xk = self.func_x = oracle.obj_func(self.x)
            oracle.update_xk(self.xk)
            self.iter += 1
            self.iter_sub = 0
            self.set_subproblem(oracle)
        else:
            self.func_x = oracle.obj_func(self.x, False)
        self.iter_sub += 1
        self.iter_sub_total += 1


class OurLM(Base):
    def __init__(
        self,
        subalg="apg",
        rho_init=1e-3,
        rho_inc=2,
        rho_dec=0.95,
        eta_inc=2,
        eta_dec=0.95,
        tol_coef_inner=0.5,
    ):
        super().__init__()
        self.subalg = subalg
        self.xk = None
        self.func_xk = None
        self.mu = None
        self.rho = rho_init
        self.rho_inc = rho_inc
        self.rho_dec = rho_dec
        self.eta = 0
        self.eta_inc = eta_inc
        self.eta_dec = eta_dec
        self.tol_coef_inner = tol_coef_inner
        self.iter_sub = 0
        self.iter_sub_total = 0

    @property
    def params(self):
        return {
            "iter_sub_total": self.iter_sub_total,
            "rho": self.rho,
            "eta": self.eta,
            "mu": self.mu,
        }

    def initialize(self, oracle: internal.Oracle, x0):
        super().initialize(oracle, x0)
        self.xk = x0
        self.func_xk = self.func_x
        oracle.update_xk(self.xk)
        self.set_subproblem(oracle)

    def set_subproblem(self, oracle: internal.Oracle):
        self.mu = self.rho * np.sqrt(self.func_xk - oracle.h_min)
        self.oracle_sub = internal.OracleSub(oracle, self.mu)
        if self.subalg == "apg":
            self.alg_sub = submethod.AcceleratedProximalGradient(
                self.x,
                self.mu,
                self.tol_coef_inner,
                self.eta,
                self.eta_inc,
                self.eta_dec,
            )
        elif self.subalg == "scipy":
            self.alg_sub = submethod.ScipyOpt(
                self.x,
                TOL_INNER_EXACT,
            )

    def update(self, oracle: internal.Oracle):
        terminated_sub = self.alg_sub.update_and_check(self.oracle_sub)
        self.x = self.alg_sub.x
        if self.subalg == "apg":
            self.eta = self.alg_sub.eta
        if terminated_sub:
            f = oracle.obj_func(self.x)
            norm_u = np.linalg.norm(self.x - self.xk)
            if f <= self.func_xk - (1 - self.tol_coef_inner) / 2 * self.mu * norm_u**2:
                self.optimality = norm_u * self.mu
                self.rho *= self.rho_dec
                self.xk = self.x
                self.func_xk = self.func_x = f
                oracle.update_xk(self.xk)
                self.iter += 1
            else:
                self.rho *= self.rho_inc
                self.x = self.xk
                self.func_x = self.func_xk
            self.iter_sub = 0
            self.set_subproblem(oracle)
        else:
            self.func_x = oracle.obj_func(self.x, False)
        self.iter_sub += 1
        self.iter_sub_total += 1
