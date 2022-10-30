import pandas as pd
from . import method, internal
import numpy as np
import pprint


class CompositeMinimization:
    algorithms = {
        "ourlm": method.OurLM,
        "pg": method.ProximalGradient,
        "dl": method.DL2018,
        "dp": method.DP2019,
    }

    def __init__(self, instance, alg_id, alg_param):
        self.instance = instance
        self.alg: method.Base = CompositeMinimization.algorithms[alg_id](**alg_param)

        self.oracle = internal.Oracle(instance)
        self.has_test = hasattr(self.instance, "test")

        self.results = []
        self.sols = []

    def __compute_result(self, tested):
        result = {
            "iter": self.alg.iter,
            "oracle_cost": self.oracle.total_cost,
            "obj_value": self.alg.func_x,
            "optimality": self.alg.optimality,
        }
        if tested:
            result |= self.instance.test(self.alg.x)
        result |= self.alg.params
        result |= self.oracle.count
        return pd.Series(result).to_frame().T

    def __store_print_result(self, tested, printed, store_sols):
        result = self.__compute_result(tested)
        if printed:
            # print(result.to_string(index=False, float_format=lambda x: "{:.3e}".format(x)))
            pprint.pprint(result.to_dict())
        self.results.append(result)
        if store_sols:
            self.sols.append(self.alg.x)

    def solve(
        self,
        max_iter=100,
        max_oracle=100,
        tol=1e-16,
        test_interval=1,
        print_interval=1,
        store_sols=False,
    ):
        self.alg.initialize(self.oracle, self.instance.x0)
        self.__store_print_result(self.has_test, True, store_sols)
        for iter in range(1, max_iter + 1):
            self.alg.update(self.oracle)
            self.__store_print_result(
                iter % test_interval == 0 and self.has_test,
                iter % print_interval == 0,
                store_sols,
            )
            # if self.oracle.total_cost >= max_oracle or (self.alg.optimality is not None and self.alg.optimality <= tol):
            if self.oracle.total_cost >= max_oracle or self.alg.func_x - self.oracle.h_min <= tol or self.alg.func_x >= np.inf:
                break

    def save_result(self, filename):
        df = pd.concat(self.results, ignore_index=True)
        df.to_csv(f"{filename}.csv", index=False)
        if self.sols:
            np.savetxt(f"sols_{filename}.txt", self.sols)
