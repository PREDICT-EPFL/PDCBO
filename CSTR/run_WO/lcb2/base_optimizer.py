"""
Implement optimizer base class.
"""
import numpy as np
import safeopt
import GPy


def get_abs_noise_level(func_arr, noise_fraction):
    obj_max = np.max(func_arr)
    obj_min = np.min(func_arr)
    obj_range = obj_max - obj_min
    obj_noise_level = obj_range * noise_fraction
    return obj_noise_level


class BaseBO:

    def __init__(self, opt_problem, base_config, reverse_meas=False):
        if 'normalize_input' in base_config.keys():
            self.normalize_input = base_config['normalize_input']
        else:
            self.normalize_input = True

        self.opt_problem = opt_problem
        noise_level = base_config['noise_level']
        if type(noise_level) == list:
            self.noise_level = base_config['noise_level']
        else:
            self.noise_level = [base_config['noise_level'] for
                                _ in range(self.opt_problem.num_constrs + 1)]

        if 'train_noise_level' in base_config.keys():
            self.train_noise_level = base_config[
                'train_noise_level']
        else:
            self.train_noise_level = 1.0  # default train noise level
        self.kernel_var = base_config['kernel_var']

        # Bounds on the input variables to be tuned
        self.bounds = opt_problem.bounds
        self.discrete_num_list = opt_problem.discretize_num_list

        # set of parameters
        self.parameter_set = safeopt.linearly_spaced_combinations(
            self.bounds,
            self.discrete_num_list
        )

        # Initial safe point
        self.x0_arr = opt_problem.init_safe_points

        # list to track the query history
        self.query_points_list = []
        self.query_points_obj = []
        self.query_points_constrs = []

        if 'kernel' in self.opt_problem.config.keys():
            self.kernel_list = self.opt_problem.config['kernel']
        elif 'kernel_type' in base_config.keys():
            self.kernel_list = self.set_kernel(kernel_type=base_config[
                'kernel_type'])
        else:
            self.kernel_list = self.set_kernel()

        print('Get initial value')
        init_obj_val_arr, init_constr_val_arr = \
            self.get_obj_constr_val(self.x0_arr)
        self.best_obj = np.min(init_obj_val_arr)
        self.init_obj_val_list = [init_obj_val_arr[0, 0]]
        self.init_constr_val_list = [init_constr_val_arr[0, :]]
        init_obj_val_arr_1d = np.squeeze(init_obj_val_arr)
        best_sol_id = np.argmin(init_obj_val_arr_1d)
        self.best_sol = self.x0_arr[best_sol_id, :]

        if reverse_meas:
            init_obj_val_arr = - init_obj_val_arr
            init_constr_val_arr = - init_constr_val_arr

        if self.normalize_input:
            self.gp_obj_mean = np.mean(init_obj_val_arr)
        else:
            self.gp_obj_mean = 0.0

        if init_obj_val_arr.ndim == 1:
            init_obj_val_arr = np.expand_dims(init_obj_val_arr, axis=1)
        self.gp_obj = GPy.models.GPRegression(
            self.x0_arr,
            init_obj_val_arr - self.gp_obj_mean,
            self.kernel_list[0],
            noise_var=self.noise_level[0] ** 2
        )

        self.gp_constr_list = []
        self.gp_constr_mean_list = []
        for i in range(self.opt_problem.num_constrs):
            if self.normalize_input:
                gp_constr_mean = np.mean(init_constr_val_arr[:, i])
            else:
                gp_constr_mean = 0.0
            self.gp_constr_list.append(
                GPy.models.GPRegression(self.x0_arr,
                                        np.expand_dims(
                                            init_constr_val_arr[:, i], axis=1)
                                        - gp_constr_mean,
                                        self.kernel_list[i + 1],
                                        noise_var=self.noise_level[i+1] ** 2))
            self.gp_constr_mean_list.append(gp_constr_mean)
        print('gps constructed.')
        self.opt = safeopt.SafeOpt([self.gp_obj] + self.gp_constr_list,
                                   self.parameter_set,
                                   [-np.inf] + [0.] *
                                   self.opt_problem.num_constrs,
                                   lipschitz=None,
                                   threshold=0.1
                                   )

        self.cumu_vio_cost = np.zeros(self.opt_problem.num_constrs)

    def get_kernel_train_noise_level(self, noise_fraction=1.0/3.0):
        train_obj = self.opt_problem.train_obj
        obj_noise_level = get_abs_noise_level(train_obj, noise_fraction)
        # set the noise level for learning hyper-parameters

        constr_noise_level_list = []
        for i in range(self.opt_problem.num_constrs):
            constr_obj = np.expand_dims(self.opt_problem.train_constr[:, i],
                                        axis=1)
            constr_noise_level = get_abs_noise_level(
                constr_obj, noise_fraction
            )
            constr_noise_level_list.append(constr_noise_level)
        return obj_noise_level, constr_noise_level_list

    def get_kernel(self, kernel_type):
        if kernel_type == 'Gaussian':
            kernel = GPy.kern.RBF(input_dim=len(self.bounds),
                                  variance=self.kernel_var,
                                  lengthscale=5.0,
                                  ARD=True)
        elif kernel_type == 'polynomial':
            kernel = GPy.kern.Poly(input_dim=len(self.bounds),
                                   variance=self.kernel_var,
                                   scale=5.0,
                                   order=4)
        return kernel

    def set_kernel(self, kernel_type='Gaussian', noise_fraction=1.0 / 2.0):
        obj_noise_level, constr_noise_level_list = \
            self.get_kernel_train_noise_level(noise_fraction)
        kernel_list = []
        opt_problem = self.opt_problem
        num_train_data, _ = opt_problem.train_obj.shape
        obj_noise = obj_noise_level * np.random.randn(
            num_train_data, 1)
        kernel = self.get_kernel(kernel_type)
        obj_gp = GPy.models.GPRegression(
                                        opt_problem.train_X,
                                        opt_problem.train_obj+obj_noise,
                                        kernel
                                        )
        obj_gp.optimize()
        kernel_list.append(kernel)
        for i in range(opt_problem.num_constrs):
            kernel_cons = self.get_kernel(kernel_type)
            constr_obj = np.expand_dims(opt_problem.train_constr[:, i],
                                        axis=1)
            constr_noise_level = constr_noise_level_list[i]
            constr_noise = constr_noise_level * np.random.randn(
                    num_train_data, 1)
            constr_gp = GPy.models.GPRegression(
                                  opt_problem.train_X,
                                  constr_obj + constr_noise,
                                  kernel_cons)
            constr_gp.optimize()
            kernel_list.append(constr_gp.kern.copy())
        return kernel_list

    def get_obj_constr_val(self, x_arr, noise=False):
        obj_val_arr, constr_val_arr = self.opt_problem.sample_point(x_arr)
        return obj_val_arr, constr_val_arr

    def plot(self):
        # Plot the GP
        self.opt.plot(100)

    def get_acquisition(self):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError

    def step_sample_point(self, update_hyperparams=False, reverse_meas=False):
        try:
            x_next = self.optimize()
        except Exception as e:
            print(f'Exception {e}.')
            x_next = self.x0_arr[0, :]
        if np.ndim(x_next) == 1:
            x_next = np.array([x_next])
        # Get a measurement of objective and constraints
        y_obj, constr_vals = self.get_obj_constr_val(x_next)

        if np.all(constr_vals <= 0):
            # update best solution and objective if we get a feasible point
            if y_obj[0, 0] < self.best_obj:
                self.best_sol = x_next
            self.best_obj = np.min([y_obj[0, 0], self.best_obj])

        # tracking violation cost
        self.query_points_list.append(x_next)
        self.query_points_obj.append(y_obj)
        self.query_points_constrs.append(constr_vals)

        print(constr_vals)
        vio_cost = self.opt_problem.get_total_violation_cost(constr_vals)
        print(vio_cost)
        violation_total_cost = np.sum(vio_cost, axis=0)
        self.cumu_vio_cost = self.cumu_vio_cost + violation_total_cost

        if reverse_meas:
            y_obj = - y_obj
            constr_vals = - constr_vals

        # Add this to the GP model
        prev_X = self.opt.gps[0].X
        prev_obj = self.opt.gps[0].Y + self.gp_obj_mean
        prev_constr_list = []
        for i in range(self.opt_problem.num_constrs):
            prev_constr_list.append(
                self.opt.gps[i + 1].Y + self.gp_constr_mean_list[i]
            )

        new_X = np.vstack([prev_X, x_next])
        new_obj = np.vstack([prev_obj, y_obj])
        if update_hyperparams:
            self.gp_obj_mean = np.mean(new_obj)
        new_obj = new_obj - self.gp_obj_mean
        self.opt.gps[0].set_XY(new_X, new_obj)
        if update_hyperparams:
            self.opt.gps[0].optimize()

        for i in range(self.opt_problem.num_constrs):
            new_constr = np.vstack(
                [prev_constr_list[i],
                 np.expand_dims(constr_vals[:, i], axis=1)]
            )
            if update_hyperparams:
                self.gp_constr_mean_list[i] = np.mean(new_constr)
            new_constr = new_constr - self.gp_constr_mean_list[i]
            self.opt.gps[i + 1].set_XY(new_X, new_constr)
            if update_hyperparams:
                self.opt.gps[i + 1].optimize()

        return x_next, y_obj, constr_vals, vio_cost

    def make_step(self, update_hyperparams=False):
        raise NotImplementedError
