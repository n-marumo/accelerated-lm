import jax.numpy as jnp
import jax
import scipy.sparse.linalg as spslin
from . import internal, submethod


class Base:
    def __init__(self):
        self.iter = 0

    @property
    def recorded_params(self):
        return {}

    @property
    def solutions(self):
        return {}

    def update(self, oracle: internal.Oracle):
        pass


class ProximalGradient(Base):
    default_params = {
        "eta_init": 1e-3,
        "eta_inc": 2,
        "eta_dec": 0.95,
    }

    def __init__(self, params, x0, oracle: internal.Oracle):
        super().__init__()
        self.params = dict(self.default_params)
        self.params.update(params)
        self.x = x0
        self.f_x, self.f_grad_x = oracle.f_value_and_grad(x0)
        self.obj_x = self.f_x + oracle.g(x0)
        self.eta = self.params["eta_init"]

    @property
    def recorded_params(self):
        return {
            "eta": self.eta,
        }

    @property
    def solutions(self):
        return [
            {"sol": self.x, "f": self.f_x, "f_grad": self.f_grad_x},
        ]

    def update(self, oracle: internal.Oracle):
        self.iter += 1
        while True:
            y = oracle.g_prox(self.x - self.f_grad_x / self.eta, 1 / self.eta)
            f_y = oracle.f(y)
            u = y - self.x
            if f_y <= self.f_x + jnp.vdot(self.f_grad_x, u) + self.eta / 2 * jnp.linalg.norm(u) ** 2:
                break
            else:
                self.eta *= self.params["eta_inc"]
        self.x = y
        self.f_x = f_y
        self.f_grad_x = oracle.f_grad(self.x)
        self.eta *= self.params["eta_dec"]


class OurLM(Base):
    default_params = {
        "subalg": "apg",
        "rho_init": 1e-3,
        "rho_inc": 2,
        "rho_dec": 0.95,
        "eta_inc": 2,
        "eta_dec": 0.95,
        "tol_coef_inner": 0.5,
        "gh_min": None,
    }

    def __init__(self, params, x0, oracle: internal.Oracle):
        super().__init__()
        self.params = dict(self.default_params)
        self.params.update(params)
        self.x = x0
        self.xk = x0
        self.obj_xk = oracle.obj(self.xk)
        self.mu = None
        self.rho = self.params["rho_init"]
        self.eta = 0

        self.gh_min = self.params["gh_min"]
        if self.gh_min is None:
            self.gh_min = oracle.gmin_plus_hmin
        print(f"gh_min: {self.gh_min}")

        self.iter_sub = 0
        self.iter_sub_total = 0
        oracle.update_xk(self.xk)
        self.set_subproblem(oracle)

    @property
    def recorded_params(self):
        return {
            "iter_sub": self.iter_sub,
            "iter_sub_total": self.iter_sub_total,
            "rho": self.rho,
            "eta": self.eta,
            "mu": self.mu,
        }

    @property
    def solutions(self):
        return [
            {"sol": self.x},
        ]

    def set_subproblem(self, oracle: internal.Oracle):
        self.mu = self.rho * jnp.sqrt(self.obj_xk - self.gh_min)
        self.oracle_sub = internal.OracleSub(oracle, self.mu)
        if self.params["subalg"] == "apg":
            self.alg_sub = submethod.AcceleratedProximalGradient(
                self.xk,
                self.mu,
                self.params["tol_coef_inner"],
                self.eta,
                self.params["eta_inc"],
                self.params["eta_dec"],
            )

    def update(self, oracle: internal.Oracle):
        self.iter_sub += 1
        self.iter_sub_total += 1
        terminated_sub = self.alg_sub.update_and_check(self.oracle_sub, self.obj_xk)
        self.x = self.alg_sub.x
        if self.params["subalg"] == "apg":
            self.eta = self.alg_sub.eta
        if terminated_sub:
            norm_u = jnp.linalg.norm(self.x - self.xk)
            obj_x = oracle.obj(self.x)
            if obj_x <= self.obj_xk - (1 - self.params["tol_coef_inner"]) / 2 * self.mu * norm_u**2:
                self.xk = self.x
                self.obj_xk = obj_x
                oracle.update_xk(self.xk)
                self.rho *= self.params["rho_dec"]
                self.iter += 1
            else:
                self.x = self.xk
                self.rho *= self.params["rho_inc"]
            self.set_subproblem(oracle)
            self.iter_sub = 0


class DP2019(Base):
    default_params = {
        "t": 1e3,
        "lip_sub": 1e0,
    }

    def __init__(self, params, x0, oracle: internal.Oracle):
        super().__init__()
        self.params = dict(self.default_params)
        self.params.update(params)
        self.x = x0
        self.xk = x0
        tlipsub = self.params["t"] * self.params["lip_sub"]
        self.iter_num_sub = 1 + jnp.sqrt(tlipsub / 2) * jnp.log(tlipsub)
        self.iter_sub = 0
        self.iter_sub_total = 0
        oracle.update_xk(self.xk)
        self.set_subproblem(oracle)

    @property
    def recorded_params(self):
        return {
            "iter_sub": self.iter_sub,
            "iter_sub_total": self.iter_sub_total,
            "mu": 1 / self.params["t"],
        }

    @property
    def solutions(self):
        return [
            {"sol": self.x},
        ]

    def set_subproblem(self, oracle: internal.Oracle):
        self.oracle_sub = internal.OracleSub(oracle, 1 / self.params["t"])
        self.alg_sub = submethod.AcceleratedProximalGradient2(
            self.x,
            self.params["lip_sub"],
            1 / self.params["t"],
            self.iter_num_sub,
        )

    def update(self, oracle: internal.Oracle):
        self.iter_sub += 1
        self.iter_sub_total += 1
        terminated_sub = self.alg_sub.update_and_check(self.oracle_sub)
        self.x = self.alg_sub.x
        if terminated_sub:
            self.xk = self.x
            oracle.update_xk(self.xk)
            self.set_subproblem(oracle)
            self.iter += 1
            self.iter_sub = 0


# https://doi.org/10.1137/21M1409536
# Algorithm 3.1
# Subproblem is constructed according to Section 4.
# assume that the proximal term is separable and employ Section 5.1
class ABO2022(Base):
    default_params = {
        "model_type": "lbfgs",
        "eta1": 0.25,
        "eta2": 0.75,
        "gamma1": 0.5,
        "gamma2": 0.5,
        "gamma3": 2,
        "gamma4": 2,
        "alpha": 1,
        "beta": 2,
        "Delta0": 1,
        "theta": 1e-3,
        "qnewton_memory": 5,
        "maxiter_sub": 1000000000,
        "subtol_delta": 1e-2,
        "subtol_coef": 1,
        "eigtol": 1e-6,
        "eigmaxiter": 10,
        "rndseed": 0,
    }

    def __init__(self, params, x0, oracle: internal.Oracle):
        super().__init__()
        self.params = dict(self.default_params)
        self.params.update(params)
        self.x = x0
        self.key = jax.random.PRNGKey(self.params["rndseed"])

        self.nu = None
        self.nu_inner = None
        self.xi = None
        self.obj_dec = None
        self.model_dec = None
        self.rho = None
        self.eigvec = jnp.ones_like(x0)
        self.Delta = self.params["Delta0"]
        self.maxiter_sub = self.params["maxiter_sub"]
        self.iter_sub = 0
        self.iter_sub_total = 0
        self.eigiter = 0
        self.eigiter_total = 0

        self.f_x, self.f_grad_x = oracle.f_value_and_grad(self.x)
        self.g_x = oracle.g(self.x)

        if self.params["model_type"] == "lbfgs":
            self.B = internal.BFGSB(n=len(x0), memory=self.params["qnewton_memory"])
        elif self.params["model_type"] == "gn":
            self.B = internal.GaussNewton(oracle=oracle)
            oracle.update_xk(self.x, ABOGN=True)

    @property
    def recorded_params(self):
        return {
            "iter_sub": self.iter_sub,
            "iter_sub_total": self.iter_sub_total,
            "eigiter": self.eigiter,
            "eigiter_total": self.eigiter_total,
            "nu": self.nu,
            "nu_inner": self.nu_inner,
            "Delta": self.Delta,
            "xi": self.xi,
            "obj_dec": self.obj_dec,
            "model_dec": self.model_dec,
            "rho": self.rho,
        }

    @property
    def solutions(self):
        return [
            {"sol": self.x, "f": self.f_x, "f_grad": self.f_grad_x, "g": self.g_x},
        ]

    def subsolver(self, oracle: internal.OracleSubABO2022, x0, nu):
        # print(f"tol: {tol}")
        # nu: step-size
        x = x0
        f_x, f_grad_x = oracle.f_value_and_grad(x)
        g_x = oracle.g(x)
        self.iter_sub = 0
        # norm_subgrad = jnp.inf
        while self.iter_sub < self.maxiter_sub:
            # print(norm_subgrad, tol)
            self.iter_sub += 1
            self.iter_sub_total += 1
            x_old = x
            f_grad_x_old = f_grad_x
            x = oracle.g_prox(x - nu * f_grad_x, nu)
            f_x, f_grad_x = oracle.f_value_and_grad(x)
            g_x = oracle.g(x)

            lhs = jnp.linalg.norm(
                f_grad_x - f_grad_x_old - (x - x_old) / nu
            )  # Eq. (4.2b), or equivalently Line 10 of Algorithm 2 of https://arxiv.org/abs/2204.12016
            rhs = min(self.params["subtol_delta"], self.xi**0.5) * self.xi  # p. 923
            # print(lhs, rhs)
            if lhs <= rhs * self.params["subtol_coef"]:
                break
        return x, f_x, g_x

    # P(Δ; x, ν) defined by Eq. (3.3b), where ψ is defined by Eq. (5.1)
    # see also the last equation in p. 917
    def P(self, oracle: internal.Oracle, Delta, x, nu, f_grad_x=None):
        if f_grad_x is None:
            f_grad_x = oracle.f_grad(x)
        return jnp.clip(
            oracle.g_prox(x - nu * f_grad_x, nu) - x,
            a_min=-Delta,
            a_max=Delta,
        )

    def max_eigval(self, A):
        # computation by scipy, which is sometimes very slow
        # n = len(self.x)
        # res = spslin.eigsh(
        #     spslin.LinearOperator((n, n), matvec=A.dot),
        #     k=1,
        #     return_eigenvectors=False,
        #     maxiter=self.params["eigmaxiter"],
        # )
        # print(res)
        # return res[0]

        # power method with random initialization
        lam = -jnp.inf
        self.key, subkey = jax.random.split(self.key)
        v = jax.random.normal(subkey, (len(self.x),))
        v /= jnp.linalg.norm(v)
        self.eigiter = 0
        for _ in range(self.params["eigmaxiter"]):
            self.eigiter += 1
            lam_pre = lam
            v_pre = v
            v = A.dot(v)
            lam = jnp.vdot(v, v_pre)
            v /= jnp.linalg.norm(v)
            # print(lam)
            if jnp.abs(lam - lam_pre) <= self.params["eigtol"] * jnp.abs(lam):
                break
        self.eigiter_total += self.eigiter
        return lam

    def update(self, oracle: internal.Oracle):
        self.iter += 1

        # Step 4
        # print("step 4")
        if self.params["model_type"] == "lbfgs" and self.B.empty:
            Lxk = 1
        else:
            Lxk = self.max_eigval(self.B)
        self.nu = 1 / (Lxk + 1 / (self.params["alpha"] * self.Delta))

        # Step 7 (with Step 6)
        # based on Section 5.1
        # print("step 7")
        s1 = self.P(
            oracle,
            Delta=self.Delta,
            x=self.x,
            nu=self.nu,
            f_grad_x=self.f_grad_x,
        )

        # compute ξ(Δ; x + s1, ν) for inner tolerance (see p. 923)
        # ξ is defined by Eq. (3.6)
        xs1 = self.x + s1
        f_grad_xs1 = oracle.f_grad(xs1)
        s2 = self.P(
            oracle,
            Delta=self.Delta,
            x=xs1,
            nu=self.nu,
            f_grad_x=f_grad_xs1,
        )
        # p(Δ; xs1, ν) = 1/(2ν) ||s2 + ν ∇f(xs1)||^2 + f(xs1) - ν/2 ||∇f(x)||^2 + h(xs1 + s2) + χ(...
        # ξ(Δ; xs1, ν) = f(xs1) + h(xs1) - p(Δ; xs1, ν)
        #              = h(xs1) - h(xs1 + s2) - 1/(2ν) ||s2 + ν ∇f(xs1)||^2 + ν/2 ||∇f(xs1)||^2
        #              = h(xs1) - h(xs1 + s2) - ∇f(xs1) @ s2 - 1/(2ν) ||s2||^2
        # Note: h -> g
        self.xi = (
            oracle.g(xs1) - oracle.g(xs1 + s2) - jnp.vdot(f_grad_xs1, s2) - jnp.linalg.norm(s2) ** 2 / (2 * self.nu)
        )

        # Step 8 (with Step 5)
        # print("step 8")
        oracle_sub = internal.OracleSubABO2022(
            oracle,
            B=self.B,
            x=self.x,
            f_grad_x=self.f_grad_x,
            Delta=min(self.Delta, self.params["beta"] * jnp.linalg.norm(s1, ord=jnp.inf)),
        )

        # set step-size for subproblem, based on Corollary 4.3
        if Lxk > 0:
            self.nu_inner = (1 - self.params["theta"]) / Lxk
        else:
            self.nu_inner = 1

        s, f_model_xs, g_xs = self.subsolver(
            oracle_sub,
            x0=s1,
            nu=self.nu_inner,
        )
        # shifted so that f_model(0) = 0

        # Step 9
        # print("step 9")
        xs = self.x + s
        self.obj_dec = self.f_x - oracle.f(xs) + self.g_x - g_xs
        self.model_dec = 0 - f_model_xs + self.g_x - g_xs
        self.rho = self.obj_dec / self.model_dec

        # Step 10
        # print("step 10")
        if self.rho >= self.params["eta1"]:
            self.x = xs
            f_grad_x_old = self.f_grad_x
            self.f_x, self.f_grad_x = oracle.f_value_and_grad(self.x)
            self.g_x = oracle.g(self.x)
            # for quasi-Newton model
            if self.params["model_type"] == "lbfgs":
                self.B.update(s, self.f_grad_x - f_grad_x_old)
            elif self.params["model_type"] == "gn":
                oracle.update_xk(self.x, ABOGN=True)

        # Step 11
        if self.rho >= self.params["eta2"]:
            self.Delta *= (self.params["gamma3"] + self.params["gamma4"]) / 2
        elif self.rho >= self.params["eta1"]:
            pass
        else:
            self.Delta *= (self.params["gamma1"] + self.params["gamma2"]) / 2
