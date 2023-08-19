import numpy as np
import GPy
import safeopt
import copy
from matplotlib import pyplot as plt
"""
Define and implement the class of optimization problem.
"""


class OptimizationProblem:

    def __init__(self, config):
        self.config = config
        self.evaluated_points_list = []
        self.evaluated_objs_list = []
        self.evaluated_constrs_list = []
        self.evaluated_feasible_points_list = []
        self.evaluated_feasible_objs_list = []
        self.evaluated_feasible_constrs_list = []

        self.problem_name = config['problem_name']

        # if obj and constr are evaluated simultaneously using a simulator
        self.eval_simu = config['eval_simu']

        if self.eval_simu:
            self.simulator = None

        self.var_dim = config['var_dim']
        self.num_constrs = config['num_constrs']
        self.obj = config['obj']
        self.constrs_list = config['constrs_list']
        self.constrs_vio_cost_funcs_list = config['vio_cost_funcs_list']
        self.vio_cost_funcs_inv_list = config['vio_cost_funcs_inv_list']
        self.bounds = config['bounds']
        self.discretize_num_list = config['discretize_num_list']
        self.init_safe_points = config['init_safe_points']
        self.train_X = config['train_X']
        self.train_obj, self.train_constr = self.sample_point(self.train_X)
        self.candidates = safeopt.\
            linearly_spaced_combinations(self.bounds, self.discretize_num_list)

    def get_minimum(self):
        obj_val, constr = self.sample_point(self.candidates)
        obj_val = obj_val.squeeze()
        feasible = np.array([True] * len(obj_val))
        for i in range(self.num_constrs):
            feasible = feasible & (constr[:, i] <= 0)

        minimum = np.min(obj_val[feasible])
        feasible_candidates = self.candidates[feasible, :]
        minimizer = feasible_candidates[np.argmin(obj_val[feasible]), :]
        return minimum, minimizer

    def sample_point(self, x, reset_init=False):
        if self.eval_simu:
            if reset_init:
                self.simulator = None
            obj_val, constraint_val_arr, simulator = self.obj(x,
                                                              self.simulator)
            self.simulator = simulator
        else:
            obj_val = self.obj(x)
            obj_val = np.expand_dims(obj_val, axis=1)
            constraint_val_list = []
            for g in self.constrs_list:
                constraint_val_list.append(g(x))
            constraint_val_arr = np.array(constraint_val_list).T
        if obj_val.ndim < 2:
            obj_val = np.expand_dims(obj_val, axis=1)
        self.evaluated_points_list.append(x)
        self.evaluated_objs_list.append(obj_val)
        self.evaluated_constrs_list.append(constraint_val_arr)

        if np.all(constraint_val_arr <= 0):
            self.evaluated_feasible_points_list.append(x)
            self.evaluated_feasible_objs_list.append(obj_val)
            self.evaluated_feasible_constrs_list.append(constraint_val_arr)
        return obj_val, constraint_val_arr

    def get_total_violation_cost(self, constraint_val_arr):
        constrs_vio_cost_list = []
        for i in range(self.num_constrs):
            cost_func = self.constrs_vio_cost_funcs_list[i]
            vio_cost = cost_func(np.maximum(constraint_val_arr[:, i], 0))
            constrs_vio_cost_list.append(vio_cost)
        return np.array(constrs_vio_cost_list).T

    def get_vio_from_cost(self, cost_budget):
        assert np.all(cost_budget >= 0)
        allowed_vio = np.zeros((1, self.num_constrs))
        for i in range(self.num_constrs):
            c_inv = self.vio_cost_funcs_inv_list[i](cost_budget[i])
            allowed_vio[0, i] = c_inv
        return allowed_vio

    def get_1d_kernel_params(self, X, Y, kernel='Gaussian'):
        if kernel == 'Gaussian':
            kernel = GPy.kern.RBF(input_dim=1,
                                  variance=1.0,
                                  lengthscale=5.0,
                                  )
            gp = GPy.models.GPRegression(
                                    np.expand_dims(X, axis=-1),
                                    np.expand_dims(Y, axis=-1),
                                    kernel
                            )
            gp.optimize()
        return gp.parameters

    def parameter_estimation(self, kernel='Gaussian', display=False):
        # dimension-wise parameter estimation
        var_dim = self.var_dim
        medium_input = [(self.bounds[i][0] + self.bounds[i][1]) * 0.5
                        for i in range(self.var_dim)]
        parameters_list = [[] for i in range(self.num_constrs+1)]
        for dim_i in range(var_dim):
            train_X_list = []
            train_objY_list = []
            train_constrY_list = [[] for i in range(self.num_constrs)]
            for k in range(self.discretize_num_list[dim_i]+1):
                # set X_i to the k-th value while setting others middle value
                X = medium_input
                X[dim_i] = self.bounds[dim_i][0] + \
                    (k/self.discretize_num_list[dim_i]) * \
                    (self.bounds[dim_i][1] - self.bounds[dim_i][0])
                objY, constrYs = self.sample_point(np.array([X]),
                                                   reset_init=True)
                train_X_list.append(copy.deepcopy(X))
                train_objY_list.append(objY)
                for m in range(self.num_constrs):
                    train_constrY_list[m].append(constrYs[:, m])

            train_X = np.array(train_X_list)
            train_objY = np.array(train_objY_list)
            train_constrY = np.array(train_constrY_list)
            if display:
                plt.figure()
                plt.plot(train_X[:, dim_i], train_objY[:, 0, 0])
                print(train_X[:, dim_i], train_objY[:, 0, 0])
                plt.title(str(dim_i))
            parameters_list[0].append(
                self.get_1d_kernel_params(
                    train_X[:, dim_i], train_objY[:, 0, 0], kernel
                )
            )
            for i in range(self.num_constrs):
                if display:
                    plt.figure()
                    plt.plot(train_X[:, dim_i], train_constrY[i, :, 0])
                    print(train_X[:, dim_i], train_constrY[i, :, 0])
                    plt.title(str(dim_i))

                parameters_list[i+1].append(
                    self.get_1d_kernel_params(train_X[:, dim_i],
                                              train_constrY[i, :, 0], kernel)
                )

        # Now estimate the noise level and prior mean
        sample_size = 50
        train_objY_list = []
        train_constrY_list = [[] for i in range(self.num_constrs)]
        for k in range(sample_size):
            if k == 0:
                objY, constrYs = self.sample_point(np.array([medium_input]),
                                                   reset_init=True)
            else:
                objY, constrYs = self.sample_point(np.array([medium_input]),
                                                   reset_init=False)
            train_objY_list.append(objY)
            for m in range(self.num_constrs):
                train_constrY_list[m].append(constrYs[:, m])
        train_objY = np.array(train_objY_list)
        train_constrY = np.array(train_constrY_list)
        self.simulator = None
        obj_mean = np.mean(train_objY, axis=0)
        obj_std = np.std(train_objY, axis=0)
        constr_mean = np.mean(train_constrY, axis=1)
        constr_std = np.std(train_constrY, axis=1)
        parameters_list.append([obj_mean, obj_std, constr_mean, constr_std])
        return parameters_list
