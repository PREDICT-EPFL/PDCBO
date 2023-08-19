"""
Implement safe Bayesian optimizer.
"""
import numpy as np
from .base_optimizer import BaseBO


class SafeBO(BaseBO):

    def __init__(self, opt_problem, safe_BO_config):
        super(SafeBO, self).__init__(opt_problem, safe_BO_config,
                                     reverse_meas=True)
        self.cumu_vio_cost = np.zeros(self.opt_problem.num_constrs)

    def optimize(self):
        try:
            x_next = self.opt.optimize()
            x_next = np.array([x_next])
        except Exception as e:
            print('Safe Opt fails to return a solution.')
            print(e)
            # if safe opt fails to return a solution, return an initial safe
            # one
            x_next = self.x0_arr[0, :]

        return x_next

    def make_step(self, update_gp=False):
        x_next, y_obj, constr_vals, vio_cost = self.step_sample_point(
            reverse_meas=True
        )
        return y_obj, constr_vals
