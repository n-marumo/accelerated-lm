# import functools
import jax
import jax.numpy as jnp

# import matplotlib.pyplot as plt
# from matplotlib.animation import ArtistAnimation


jax.config.update("jax_enable_x64", True)


# class Problem:
#     gmin_plus_hmin = 0

#     def __init__(self, T, N, num_obs, coef, mu, nu, sigmab=0.2, sigmao=0.05, init="zero", seed=0):
#         key = jax.random.PRNGKey(seed)
#         self.T = T
#         self.N = N
#         self.coef = coef
#         self.mu = mu
#         self.nu = nu
#         self.sigmab = sigmab
#         self.sigmao = sigmao

#         z = jnp.linspace(0, 1, num=N + 1)
#         self.x_true = jnp.sin(6 * jnp.pi * z)
#         left = 4 * N // 10
#         right = N // 2
#         self.x_true = self.x_true.at[left:right].add(
#             1.0 * (1 - jnp.cos(2 * jnp.pi * jnp.linspace(0, 1, num=right - left)))
#         )
#         self.x_true = self.x_true.at[1:-1].get()

#         self.U_true = self.__solve_pde(self.x_true)

#         key, subkey = jax.random.split(key)
#         self.y = self.x_true + jax.random.normal(subkey, (N - 1,)) * sigmab

#         if init == "zero":
#             self.x0 = jnp.zeros((N - 1,))
#         elif init == "obs":
#             self.x0 = self.y

#         key, *subkeys = jax.random.split(key, num=3)
#         self.obs_idx = jax.lax.sort(jax.random.choice(subkeys[0], jnp.arange(T * (N - 1)), (num_obs,), replace=False))
#         self.obs_val = self.U_true[self.obs_idx] + jax.random.normal(subkeys[1], (num_obs,)) * sigmao

#         # print(self.x_true)
#         # # animation
#         # fig = plt.figure(figsize=(5, 5), facecolor="lightblue")
#         # U = self.__solve_pde(self.x_true).reshape((T, -1))
#         # frames = []
#         # for t in range(self.T):
#         #     frames.append(plt.plot(z[1:-1], U[t, :], color="blue"))
#         # ani = ArtistAnimation(fig, frames, interval=20)
#         # plt.show()

#     # @functools.partial(jax.jit, static_argnums=(0,))
#     def __solve_pde(self, u_init):
#         U = jnp.zeros((self.T + 1, self.N + 1))
#         U = U.at[0:2, 1:-1].set(u_init)
#         for t in range(1, self.T):
#             U = U.at[t + 1, 1:-1].set(
#                 2 * U[t, 1:-1]
#                 - U[t - 1, 1:-1]
#                 + (self.N**2 * (U[t, :-2] - 2 * U[t, 1:-1] + U[t, 2:]) - self.nu * jnp.exp(self.nu * U[t, 1:-1]))
#                 / self.T**2
#                 * self.coef**2
#             )
#         return U[1:, 1:-1].reshape(-1)

#     # @functools.partial(jax.jit, static_argnums=(0,))
#     def c(self, x):
#         return jnp.concatenate((self.y - x, self.obs_val - self.__solve_pde(x)[self.obs_idx]))

#     # @functools.partial(jax.jit, static_argnums=(0,))
#     def h(self, r):
#         return (
#             jnp.sum(jnp.square(r[: self.N - 1])) / self.sigmab**2
#             + jnp.sum(jnp.square(r[self.N - 1 :])) / self.sigmao**2
#         ) / 2

#     def g(self, x):
#         return 0

#     def g_prox(self, x, eta):
#         return x


class Problem:
    gmin_plus_hmin = 0

    def __init__(self, T, N, num_obs, coef, mu, nu, sigmao, seed=0):
        key = jax.random.PRNGKey(seed)
        self.T = T
        self.N = N
        self.coef = coef
        self.mu = mu
        self.nu = nu
        self.x0 = jnp.zeros((N - 1,))

        z = jnp.linspace(0, 1, num=N + 1)
        self.x_true = jnp.sin(6 * jnp.pi * z)
        left = 4 * N // 10
        right = N // 2
        self.x_true = self.x_true.at[left:right].add(
            1.0 * (1 - jnp.cos(2 * jnp.pi * jnp.linspace(0, 1, num=right - left)))
        )
        self.x_true = self.x_true.at[1:-1].get()
        self.U_true = self.__solve_pde(self.x_true)

        key, *subkeys = jax.random.split(key, num=3)
        self.obs_idx = jax.lax.sort(jax.random.choice(subkeys[0], jnp.arange(T * (N - 1)), (num_obs,), replace=False))
        self.obs_val = self.U_true[self.obs_idx] + jax.random.normal(subkeys[1], (num_obs,)) * sigmao

        # print(self.x_true)
        # # animation
        # fig = plt.figure(figsize=(5, 5), facecolor="lightblue")
        # U = self.__solve_pde(self.x_true).reshape((T, -1))
        # frames = []
        # for t in range(self.T):
        #     frames.append(plt.plot(z[1:-1], U[t, :], color="blue"))
        # ani = ArtistAnimation(fig, frames, interval=20)
        # plt.show()

    def __solve_pde(self, u_init):
        U = jnp.zeros((self.T + 1, self.N + 1))
        U = U.at[0:2, 1:-1].set(u_init)
        for t in range(1, self.T):
            U = U.at[t + 1, 1:-1].set(
                2 * U[t, 1:-1]
                - U[t - 1, 1:-1]
                + (self.N**2 * (U[t, :-2] - 2 * U[t, 1:-1] + U[t, 2:]) - self.nu * jnp.exp(self.nu * U[t, 1:-1]))
                / self.T**2
                * self.coef**2
            )
        return U[1:, 1:-1].reshape(-1)

    def c(self, x):
        return self.obs_val - self.__solve_pde(x)[self.obs_idx]

    def h(self, r):
        return jnp.mean(jnp.square(r)) / 2

    def g(self, x):
        return 0

    def g_prox(self, x, eta):
        return x


# N = 2**2
# n = N - 1

# p = Problem(T=2**8, N=2**6, num_obs=2**7, coef=1e-0, mu=0, nu=1e0, sigmao=1e0)
# p = Problem(T=2**3, N=N, num_obs=2**4, coef=1e-2, mu=5e-1, nu=1e0)


# p = Problem(T=2**8, N=2**6, num_obs=2**10, coef=1e-0, mu=5e-1, nu=1e0)


# def f(x):
#     return p.h(p.c(x))


# def f_grad(x):
#     return jax.grad(f)(x)


# # x = np.random.randn(n)
# x = jnp.array([-1.2916388, -0.40940887, 0.71514289])
# print(x)
# print(p.c(x))
# print(f(x))
# print(f_grad(x))
