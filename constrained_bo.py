"""
Implement constrained Bayesian optimizer for our test.
"""
import numpy as np
import safeopt
import GPy
from scipy.stats import norm


class ConstrainedBO:

    def __init__(self, opt_problem, constrained_BO_config):
        # optimization problem and measurement noise
        self.num_eps = 1e-10
        self.opt_problem = opt_problem
        self.noise_level = constrained_BO_config['noise_level']
        if 'train_noise_level' in constrained_BO_config.keys():
            self.train_noise_level = constrained_BO_config['train_noise_level']
        else:
            self.train_noise_level = 10.0

        self.kernel_var = constrained_BO_config['kernel_var']

        # Bounds on the inputs variable
        self.bounds = opt_problem.bounds
        self.discret_num_list = opt_problem.discretize_num_list

        if 'kernel_type' in constrained_BO_config.keys():
            self.set_kernel(kernel_type=constrained_BO_config['kernel_type'])
        else:
            self.set_kernel()

        # set of parameters
        self.parameter_set = safeopt.linearly_spaced_combinations(
            self.bounds,
            self.discret_num_list
        )

        # Initial safe point
        self.x0_arr = opt_problem.init_safe_points
        self.setup_optimizer()

        safe_obj_arr, safe_constr_arr = opt_problem.sample_point(self.x0_arr)
        self.query_points_list = []
        self.query_point_obj = []
        self.query_point_constrs = []

    def get_kernel_train_noise_level(self, noise_fraction=1.0/2.0):
        obj_max = np.max(self.opt_problem.train_obj)
        obj_min = np.min(self.opt_problem.train_obj)
        obj_range = obj_max - obj_min
        obj_noise_level = obj_range * noise_fraction
        constr_noise_level_list = []
        for i in range(self.opt_problem.num_constrs):
            constr_obj = np.expand_dims(self.opt_problem.train_constr[:, i],
                                        axis=1)
            constr_max = np.max(constr_obj)
            constr_min = np.min(constr_obj)
            constr_range = constr_max - constr_min
            constr_noise_level = constr_range * noise_fraction
            constr_noise_level_list.append(constr_noise_level)
        return obj_noise_level, constr_noise_level_list

    def set_kernel(self, kernel_type='Gaussian'):
        if 'kernel' in self.opt_problem.config.keys():
            # print('kernel in opt problem config.')
            self.kernel_list = self.opt_problem.config['kernel']
            return 0

        noise_fraction = 1.0 / 2.0
        obj_noise_level, constr_noise_level_list = \
            self.get_kernel_train_noise_level(noise_fraction)
        if kernel_type == 'Gaussian':
            kernel_list = []
            kernel = GPy.kern.RBF(input_dim=len(self.bounds),
                                  variance=self.kernel_var,
                                  lengthscale=5.0,
                                  ARD=True)
            opt_problem = self.opt_problem
            num_train_data, _ = opt_problem.train_obj.shape

            obj_max = np.max(opt_problem.train_obj)
            obj_min = np.min(opt_problem.train_obj)
            obj_range = obj_max - obj_min
            obj_noise_level = obj_range * noise_fraction

            obj_noise = obj_noise_level * np.random.randn(
                num_train_data, 1)
            obj_gp = GPy.models.GPRegression(
                                             opt_problem.train_X,
                                             opt_problem.train_obj+obj_noise,
                                             kernel
                                             )
            obj_gp.optimize()
            kernel_list.append(kernel)
            for i in range(opt_problem.num_constrs):
                kernel_cons = GPy.kern.RBF(input_dim=len(self.bounds),
                                           variance=self.kernel_var,
                                           lengthscale=5.0,
                                           ARD=True)
                constr_obj = np.expand_dims(opt_problem.train_constr[:, i],
                                            axis=1)

                constr_max = np.max(constr_obj)
                constr_min = np.min(constr_obj)
                constr_range = constr_max - constr_min
                constr_noise_level = constr_range * noise_fraction

                constr_noise = constr_noise_level * np.random.randn(
                    num_train_data, 1)
                constr_gp = GPy.models.GPRegression(
                                  opt_problem.train_X,
                                  constr_obj + constr_noise,
                                  kernel_cons)
                constr_gp.optimize()
                kernel_list.append(constr_gp.kern.copy())
            self.kernel_list = kernel_list

        if kernel_type == 'polynomial':
            kernel_list = []
            kernel = GPy.kern.Poly(input_dim=len(self.bounds),
                                   variance=self.kernel_var,
                                   scale=5.0,
                                   order=4)
            opt_problem = self.opt_problem
            num_train_data, _ = opt_problem.train_obj.shape
            obj_noise = obj_noise_level * np.random.randn(
                num_train_data, 1)
            obj_gp = GPy.models.GPRegression(
                                             opt_problem.train_X,
                                             opt_problem.train_obj+obj_noise,
                                             kernel
                                             )
            obj_gp.optimize()
            kernel_list.append(kernel)
            for i in range(opt_problem.num_constrs):
                kernel_cons = GPy.kern.Poly(input_dim=len(self.bounds),
                                            variance=self.kernel_var,
                                            scale=5.0,
                                            order=1)
                constr_obj = np.expand_dims(opt_problem.train_constr[:, i],
                                            axis=1)

                constr_noise = constr_noise_level_list[i] * np.random.randn(
                    num_train_data, 1)
                constr_gp = GPy.models.GPRegression(
                                  opt_problem.train_X,
                                  constr_obj + constr_noise,
                                  kernel_cons)
                constr_gp.optimize()
                kernel_list.append(constr_gp.kern.copy())
            self.kernel_list = kernel_list

    def setup_optimizer(self):
        # The statistical model of our objective function and safety constraint
        init_obj_val_arr, init_constr_val_arr = \
            self.get_obj_constr_val(self.x0_arr)

        self.best_obj = np.min(init_obj_val_arr)
        best_obj_id = np.argmin(init_obj_val_arr[:, 0])
        self.best_sol = self.x0_arr[best_obj_id, :]

        self.gp_obj = GPy.models.GPRegression(self.x0_arr,
                                              init_obj_val_arr,
                                              self.kernel_list[0],
                                              noise_var=self.noise_level ** 2)
        self.gp_constr_list = []
        for i in range(self.opt_problem.num_constrs):
            self.gp_constr_list.append(
                GPy.models.GPRegression(self.x0_arr,
                                        np.expand_dims(
                                            init_constr_val_arr[:, i], axis=1),
                                        self.kernel_list[i+1],
                                        noise_var=self.noise_level ** 2))

        self.opt = safeopt.SafeOpt([self.gp_obj] + self.gp_constr_list,
                                   self.parameter_set,
                                   [-np.inf] + [0.] *
                                   self.opt_problem.num_constrs,
                                   lipschitz=None,
                                   threshold=0.1
                                   )
        self.cumu_vio_cost = np.zeros(self.opt_problem.num_constrs)

    def get_obj_constr_val(self, x_arr, noise=False):
        obj_val_arr, constr_val_arr = self.opt_problem.sample_point(x_arr)
        return obj_val_arr, constr_val_arr

    def plot(self):
        # Plot the GP
        self.opt.plot(100)
        # Plot the true function
        y, constr_val = self.get_obj_constr_val(self.parameter_set,
                                                noise=False)

    def get_acquisition(self, type='constrained_EI'):
        obj_mean, obj_var = self.gp_obj.predict(self.parameter_set)
        obj_mean = obj_mean.squeeze()
        obj_var = obj_var.squeeze()
        constrain_mean_list = []
        constrain_var_list = []
        for i in range(self.opt_problem.num_constrs):
            mean, var = self.gp_constr_list[i].predict(self.parameter_set)

            constrain_mean_list.append(np.squeeze(mean))
            constrain_var_list.append(np.squeeze(var))
        constrain_mean_arr = np.array(constrain_mean_list).T
        constrain_var_arr = np.array(constrain_var_list).T

        # calculate Pr(g_i(x)<=0)
        prob_negtive = norm.cdf(0, constrain_mean_arr, constrain_var_arr)
        # calculate feasibility prob
        prob_feasible = np.prod(prob_negtive, axis=1)

        # calculate EI
        f_min = self.best_obj
        z = (f_min - obj_mean)/np.maximum(np.sqrt(obj_var), self.num_eps)
        EI = (f_min - obj_mean) * norm.cdf(z) + np.sqrt(obj_var) * norm.pdf(z)
        EIc = prob_feasible * EI
        if type == 'constrained_EI':
            return EIc

    def optimize(self, type='constrained_EI'):
        if type == 'constrained_EI':
            acq = self.get_acquisition()
        next_point_id = np.argmax(acq)
        next_point = self.parameter_set[next_point_id]
        return next_point

    def make_step(self):
        x_next = self.optimize()
        x_next = np.array([x_next])

        INFINITY = 1e10
        FEASIBLE_RADIUS = 0.1
        # Get a measurement from the real system
        y_obj, constr_vals = self.get_obj_constr_val(x_next)

        self.query_points_list.append(x_next)
        self.query_point_obj.append(y_obj)
        self.query_point_constrs.append(constr_vals)

        if y_obj[0, 0] ** 2 > INFINITY:
            distance = np.linalg.norm(self.parameter_set - x_next[0, :],
                                      axis=1)
            self.parameter_set = self.parameter_set[
                (distance > FEASIBLE_RADIUS), :]
            return y_obj, constr_vals

        if np.all(constr_vals <= 0):
            if self.best_obj > y_obj[0, 0]:
                self.best_sol = x_next
            # update best objective if we get a feasible point
            self.best_obj = np.min([y_obj[0, 0], self.best_obj])
        y_meas = np.hstack((y_obj, constr_vals))
        violation_cost = self.opt_problem.get_total_violation_cost(constr_vals)
        violation_total_cost = np.sum(violation_cost, axis=0)
        self.cumu_vio_cost = self.cumu_vio_cost + violation_total_cost
        # Add this to the GP model
        self.opt.add_new_data_point(x_next, y_meas)
        return y_obj, constr_vals
