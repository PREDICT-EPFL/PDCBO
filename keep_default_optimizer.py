"""
Implement violation-aware Bayesian optimizer.
"""
import numpy as np
import safeopt
import GPy
from scipy.stats import norm


class KeepDefaultOpt:

    def __init__(self, opt_problem, keep_default_config):

        self.current_step = 0
        self.opt_problem = opt_problem
        self.noise_level = keep_default_config['noise_level']
        self.current_step = 0
        self.cumu_vio_cost = 0
        # Bounds on the inputs variable
        self.bounds = opt_problem.bounds
        self.discret_num_list = opt_problem.discretize_num_list

        # set of parameters
        self.parameter_set = safeopt.linearly_spaced_combinations(
            self.bounds,
            self.discret_num_list
        )

        # Initial safe point
        self.x0_arr = opt_problem.init_safe_points
        self.query_points_list = []
        self.query_point_obj = []
        self.query_point_constrs = []
        self.S = []
        # self.kernel_list = []
        init_obj_val_arr, init_constr_val_arr = \
            self.get_obj_constr_val(self.x0_arr)
        self.init_obj_val_arr = init_obj_val_arr
        self.init_constr_val_arr = init_constr_val_arr
        self.best_obj = np.min(init_obj_val_arr)
        best_obj_id = np.argmin(init_obj_val_arr[:, 0])
        self.best_sol = self.x0_arr[best_obj_id, :]

    def get_obj_constr_val(self, x_arr, noise=False):
        obj_val_arr, constr_val_arr = self.opt_problem.sample_point(x_arr)
        return obj_val_arr, constr_val_arr

    def plot(self):
        # Plot the GP
        self.opt.plot(100)
        # Plot the true function
        y, constr_val = self.get_obj_constr_val(self.parameter_set,
                                                noise=False)

    def make_step(self, evaluate_point=None):
        self.current_step += 1
        if evaluate_point is None:
            x_next = np.expand_dims(self.x0_arr[0, :], axis=0)
        else:
            x_next = evaluate_point
        # Get a measurement from the real system
        y_obj, constr_vals = self.get_obj_constr_val(x_next)

        self.query_points_list.append(x_next)
        self.query_point_obj.append(y_obj)
        self.query_point_constrs.append(constr_vals)

        vio_cost = self.opt_problem.get_total_violation_cost(constr_vals)
        vio_cost = np.squeeze(vio_cost)
        if np.all(constr_vals <= 0):
            # update best objective if we get a feasible point
            if self.best_obj > y_obj[0, 0]:
                self.best_sol = x_next
            self.best_obj = np.min([y_obj[0, 0], self.best_obj])
        violation_cost = self.opt_problem.get_total_violation_cost(constr_vals)
        violation_total_cost = np.sum(violation_cost, axis=0)
        self.cumu_vio_cost = self.cumu_vio_cost + violation_total_cost

        return y_obj, constr_vals
