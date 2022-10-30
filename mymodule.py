import functools
import time

import jax.numpy as jnp
import numpy as np

TIMER_ITER = 100


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kargs):
        print(f"Executing: {func.__name__}")
        start = time.time()
        for _ in range(TIMER_ITER):
            result = func(*args, **kargs)
        elapsed_time = (time.time() - start) / TIMER_ITER
        print(f"Done: {elapsed_time} [sec]")
        return result

    return wrapper


def func_name(func):
    def wrapper(*args, **kargs):
        print("####################################")
        print(f"# {func.__name__}")
        print("####################################")
        return func(*args, **kargs)

    return wrapper


def projection_l1_ball_jax(x, r):
    y = jnp.sort(jnp.abs(x))[::-1]
    z = (jnp.cumsum(y) - r) / jnp.arange(1, jnp.size(x) + 1)
    k = jnp.where(y > z)[0][-1]
    lam = z[k]

    idx_pos = x > lam
    idx_neg = x < -lam

    res = jnp.zeros_like(x)
    res = res.at[idx_pos].set(x[idx_pos] - lam)
    res = res.at[idx_neg].set(x[idx_neg] + lam)
    return res


def projection_l1_ball(x, r):
    y = np.sort(np.abs(x))[::-1]
    z = (np.cumsum(y) - r) / np.arange(1, np.size(x) + 1)
    k = np.where(y > z)[0][-1]
    lam = z[k]

    idx_pos = x > lam
    idx_neg = x < -lam

    res = np.zeros(np.size(x))
    res[idx_pos] = x[idx_pos] - lam
    res[idx_neg] = x[idx_neg] + lam
    return res
