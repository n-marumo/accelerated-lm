import pandas as pd
from . import method, internal
import numpy as np
import jax.numpy as jnp
import pprint


import time
import os

DIVERGENCE_RATIO = 1e5  # used for checking divergence
STEP_GMN = 1e5  # GMN: gradient mapping norm


class CompositeMinimization:
    algorithms = {
        "ourlm": method.OurLM,
        "pg": method.ProximalGradient,
        "dp": method.DP2019,
        # "abo": method.ABO2022OLD,
        "abo": method.ABO2022,
    }

    def __init__(self, instance):
        self.instance = instance
        self.alg: method.Base = None
        self.oracle = internal.Oracle(instance)
        self.elapsed_time = None
        self.store_sols = None
        self.results = None
        self.sols = None

    def __calc_obj_gmn(self):
        obj_gmn = []
        for sol in self.alg.solutions:
            f = None
            f_grad = None
            if "f" in sol:
                f = sol["f"]
            if "f_grad" in sol:
                f_grad = sol["f_grad"]
            if f_grad is None:
                f, f_grad = self.oracle.f_value_and_grad(sol["sol"], False)
            if f is None:
                f = self.oracle.f(sol["sol"], False)

            g = None
            if "g" in sol:
                g = sol["g"]
            if g is None:
                g = self.oracle.g(sol["sol"], False)

            obj_gmn.append(
                (
                    f + g,
                    jnp.linalg.norm(sol["sol"] - self.oracle.g_prox(sol["sol"] - STEP_GMN * f_grad, STEP_GMN, False))
                    / STEP_GMN,
                )
            )
        return tuple(map(min, zip(*obj_gmn)))

    def __store_print_result(self, obj, gmn, printed):
        result = {
            "iter": self.alg.iter,
            "elapsed_time": self.elapsed_time,
            "obj": obj,
            "gmn": gmn,
        }
        result |= self.oracle.count
        result |= self.alg.recorded_params
        result = pd.Series(result).to_frame().T
        if printed:
            pprint.pprint(result.to_dict())
        self.results.append(result)
        if self.store_sols:
            self.sols.append(self.alg.solutions[0]["sol"])

    def solve(
        self,
        alg_id,
        alg_param={},
        max_iter=100,
        timeout=20,
        tol_obj=0,
        tol_grad=0,
        print_interval=1,
        store_sols=False,
    ):
        self.store_sols = store_sols
        self.elapsed_time = 0
        self.oracle.reset_count()
        self.results = []
        self.sols = []
        self.alg: method.Base = CompositeMinimization.algorithms[alg_id](alg_param, self.instance.x0, self.oracle)
        obj, gmn = self.__calc_obj_gmn()
        self.__store_print_result(obj, gmn, True)
        obj_init = obj

        for iter in range(1, max_iter + 1):
            start_time = time.perf_counter()
            self.alg.update(self.oracle)
            end_time = time.perf_counter()
            self.elapsed_time += end_time - start_time

            obj, gmn = self.__calc_obj_gmn()
            self.__store_print_result(obj, gmn, iter % print_interval == 0)
            if obj / obj_init >= DIVERGENCE_RATIO or self.elapsed_time >= timeout or gmn <= tol_grad or obj <= tol_obj:
                break

    def save_result(self, folder, filename):
        os.makedirs(folder, exist_ok=True)
        df = pd.concat(self.results, ignore_index=True)
        df.to_csv(f"{folder}/{filename}.csv", index=False)
        if self.sols:
            np.savetxt(f"{folder}/sols_{filename}.txt", self.sols)
