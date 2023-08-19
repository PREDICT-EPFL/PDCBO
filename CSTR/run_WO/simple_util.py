import numpy as np
import math
import GPy
import safeopt
import sys
sys.path.append('..')

from sub_uts.systems import *
from sub_uts.utilities_2 import *


"""
Define some utility functions for the test of safe Bayesian optimization,
constrained Bayesian optimization, and our method.
"""


def get_config(problem_name, problem_dim=None, gp_kernel=None, init_points_id=0):
    """
    Input: problem_name
    Output: configuration of the constrained problem, including variable
    dimension, number of constraints, objective function and constraint
    function.
    """
    config = dict()
    cost_funcs = {
        'square': lambda x: np.square(x),
        'exp': lambda x: np.exp(x) - 1,
        'linear': lambda x: x
    }
    cost_funcs_inv = {
        'square': lambda x: np.sqrt(x),
        'exp': lambda x: np.log(x+1),
        'linear': lambda x: x
    }
    config['problem_name'] = problem_name

    if problem_name == 'sinusodal':
        config['var_dim'] = 2
        config['discretize_num_list'] = [100 for _ in range(config['var_dim'])]
        config['num_constrs'] = 1
        config['bounds'] = [(0, 6), (0, 6)]
        config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            [20 for _ in range(config['var_dim'])]
        )

        def f(x_f):
            return np.cos(2 * x_f[:, 0]) * np.cos(x_f[:, 1]) + np.sin(x_f[:, 0])

        def g_1(x_g1):
            return np.cos(x_g1[:, 0]) * np.cos(x_g1[:, 1]) - \
                   np.sin(x_g1[:, 0]) * np.sin(x_g1[:, 1]) + 0.2

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['vio_cost_funcs_list'] = [cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]
        config['init_safe_points'] = np.array([[math.pi * 0.5, math.pi * 0.5]])

    if problem_name == 'sinusodal_1d':
        ALPHA = 1
        config['var_dim'] = 1
        config['discretize_num_list'] = [100 * ALPHA
                                         for _ in range(config['var_dim'])]
        config['num_constrs'] = 1
        config['bounds'] = [(0, ALPHA * np.pi)]
        config['train_X'] = safeopt.linearly_spaced_combinations(
            [(0, ALPHA * np.pi/6)],
            [20 * ALPHA for _ in range(config['var_dim'])]
        )
        f = lambda x: -np.sin(x[:, 0] / ALPHA)
        g_1 = lambda x: 0.0-np.cos(x[:, 0] / ALPHA)

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['vio_cost_funcs_list'] = [cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]
        config['init_safe_points'] = np.array([[0.0]]) * ALPHA


    # Generate function with safe initial point at x=0
    def sample_safe_fun(kernel, config, noise_var, gp_kernel, safe_margin):
        while True:
            fun = safeopt.sample_gp_function(kernel, config['bounds'],
                                         noise_var, 100)
            if gp_kernel == 'Gaussian':
                if fun(0, noise=False) < -safe_margin:
                    break
            if gp_kernel == 'poly':
                if fun(0, noise=False) < -safe_margin:
                    break
        return fun

    if problem_name == 'GP_sample_single_func':
        if problem_dim is None:
            problem_dim = 1
        if gp_kernel is None:
            gp_kernel = 'Gaussian'
        config['var_dim'] = problem_dim
        config['discretize_num_list'] = [100] * problem_dim
        config['num_constrs'] = 1
        config['bounds'] = [(-10, 10)] * problem_dim
        config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            config['discretize_num_list']
        )

        # Measurement noise
        noise_var = 0.00

        # Bounds on the inputs variable
        parameter_set = \
            safeopt.linearly_spaced_combinations(config['bounds'],
                                                 config['discretize_num_list'])

        # Define Kernel
        if gp_kernel == 'Gaussian':
            kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=2.,
                                  lengthscale=1.0, ARD=True)
        if gp_kernel == 'poly':
            kernel = GPy.kern.Poly(input_dim=len(config['bounds']),
                                   variance=2.0,
                                   scale=1.0,
                                   order=1)

        # Initial safe point
        x0 = np.zeros((1, len(config['bounds'])))

        safe_margin = 0.2
        func = sample_safe_fun(kernel, config, noise_var, gp_kernel, safe_margin)
        func_min = np.min(func(parameter_set))
        config['f_min'] = func_min
        f = lambda x: func(x, noise=False).squeeze(axis=1)
        g_1 = lambda x: func(x, noise=False).squeeze(axis=1)

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['vio_cost_funcs_list'] = [cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]
        config['init_safe_points'] = x0
        config['kernel'] = [kernel, kernel.copy()]

    if problem_name == 'GP_sample_two_funcs':
        if problem_dim is None:
            problem_dim = 1
        if gp_kernel is None:
            gp_kernel = 'Gaussian'
        config['var_dim'] = problem_dim
        config['discretize_num_list'] = [100] * problem_dim
        config['num_constrs'] = 1
        config['bounds'] = [(-10, 10)] * problem_dim
        config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            config['discretize_num_list']
        )

        config['eval_simu'] = False
        # Measurement noise
        noise_var = 0.00  # 0.05 ** 2

        # Bounds on the inputs variable
        parameter_set = \
            safeopt.linearly_spaced_combinations(config['bounds'],
                                                 config['discretize_num_list'])

        # Define Kernel
        if gp_kernel == 'Gaussian':
            kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=2.,
                                  lengthscale=1.0, ARD=True)
        if gp_kernel == 'poly':
            kernel = GPy.kern.Poly(input_dim=len(config['bounds']),
                                   variance=2.0,
                                   scale=1.0,
                                   order=1)

        # Initial safe point
        x0 = np.zeros((1, len(config['bounds'])))

        constr_func = sample_safe_fun(kernel, config, noise_var, gp_kernel, safe_margin=0.2)
        obj_func = safeopt.sample_gp_function(kernel, config['bounds'],
                                                 noise_var, 100)
        func_feasible_min = np.min(obj_func(parameter_set)[constr_func(parameter_set)<=0])
        config['f_min'] = func_feasible_min
        f = lambda x: obj_func(x, noise=False).squeeze(axis=1)
        g_1 = lambda x: constr_func(x, noise=False).squeeze(axis=1)

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['vio_cost_funcs_list'] = [cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]
        config['init_safe_points'] = x0
        config['kernel'] = [kernel, kernel.copy()]

    if problem_name == 'energym_apartment_therm_tune':
        if problem_dim is None:
            problem_dim = 2
        if gp_kernel is None:
            gp_kernel = 'Gaussian'
        config['var_dim'] = problem_dim
        config['discretize_num_list'] = [100] * problem_dim
        config['num_constrs'] = 1
        config['bounds'] = [(0.05, 0.45), (0.5, 0.95)] #[(-10, 10)] * problem_dim
        config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            2 #config['discretize_num_list']
        )

        config['eval_simu'] = True
        config['eta_func'] = lambda t: 3/np.sqrt(t * 1.0)
        # Measurement noise
        noise_var = 0.00  # 0.05 ** 2

        # Bounds on the inputs variable
        parameter_set = \
            safeopt.linearly_spaced_combinations(config['bounds'],
                                                 config['discretize_num_list'])

        # Define Kernel
        if gp_kernel == 'Gaussian':
            kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=1.,
                                  lengthscale=[0.3, 0.3], ARD=True)
            constr_kernel = GPy.kern.RBF(
                input_dim=len(config['bounds']), variance=1.,
                lengthscale=[0.3, 2.0], ARD=True)

        if gp_kernel == 'poly':
            kernel = GPy.kern.Poly(input_dim=len(config['bounds']),
                                   variance=2.0,
                                   scale=1.0,
                                   order=1)

        # Initial safe point
        x0 = np.array([[0.13, 0.9]])

        def f(x, simulator_to_use=None):
            energy_mean = 385.3506855413606
            energy_std = 15.832128159177238
            dev_mean = 0.48119964518630765
            dev_std = 0.016940298884339722
            size_batch, _ = x.shape
            energy_list = []
            dev_list = []
            #print(f'Size batch {size_batch}!')
            for k in range(size_batch):
                energy, dev = get_ApartTherm_kpis(x[k, 0], x[k, 1])
                energy_list.append(energy)
                dev_list.append(dev)
            energy_arr = (np.array(energy_list) - energy_mean) / energy_std
            dev_arr = (np.array(dev_list) - dev_mean) / dev_std
            return energy_arr, simulator_to_use, dev_arr

        def g_1(x, simulator_to_use=None):
            energy_mean = 385.3506855413606
            energy_std = 15.832128159177238
            dev_mean = 0.48119964518630765
            dev_std = 0.016940298884339722
            size_batch, _ = x.shape
            energy_list = []
            dev_list = []
            for k in range(size_batch):
                energy, dev = get_ApartTherm_kpis(x[k, 0], x[k, 1])
                energy_list.append(energy)
                dev_list.append(dev)
            energy_arr = (np.array(energy_list) - energy_mean) / energy_std
            dev_arr = (np.array(dev_list) - dev_mean) / dev_std
            return dev_arr, simulator_to_use, energy_arr

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['vio_cost_funcs_list'] = [cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]
        config['init_safe_points'] = x0
        config['kernel'] = [kernel, constr_kernel]
        print(config)

    if problem_name == 'WO':
        if problem_dim is None:
            problem_dim = 2
        if gp_kernel is None:
            gp_kernel = 'Gaussian'
        config['var_dim'] = problem_dim
        config['discretize_num_list'] = [100] * problem_dim
        config['num_constrs'] = 2
        config['bounds'] = [(4, 7), (70, 100)] #[(-10, 10)] * problem_dim
        config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            2 #config['discretize_num_list']
        )

        config['eval_simu'] = False
        config['eta_func'] = lambda t: 3/np.sqrt(t * 1.0)
        # Measurement noise
        noise_var = 0.00  # 0.05 ** 2

        # Bounds on the inputs variable
        parameter_set = \
            safeopt.linearly_spaced_combinations(config['bounds'],
                                                 config['discretize_num_list'])

        constr_1_std = 0.03 * 2
        constr_2_std = 4e-4 * 2 * 10
        # Define Kernel
        if gp_kernel == 'Gaussian':
            kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=1.,
                                  lengthscale=[1.0, 10.0], ARD=True)
            constr_kernel_1 = GPy.kern.RBF(
                input_dim=len(config['bounds']), variance= 1.0, #(0.03 * 2)**2,
                lengthscale=[1.0, 10.0], ARD=True)
            constr_kernel_2 = GPy.kern.RBF(
                input_dim=len(config['bounds']), variance= 1.0,  #(4e-4 * 2) **2,
                lengthscale=[1.0, 10.0], ARD=True)

        # Initial safe point
        x0 = np.array([[4.3, 80.0]]) # [[6.9, 83]])

        model = WO_model()
        plant = WO_system()

        obj_model = model.WO_obj_ca
        cons_model = [model.WO_con1_model_ca, model.WO_con2_model_ca]
        obj_system = plant.WO_obj_sys_ca
        cons_system = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]

        def f(x):
            obj_mean = -52.76428219620413
            obj_std = 22.763438402688305
            size_batch, _ = x.shape
            obj_list = []
            for k in range(size_batch):
                obj_val = float(obj_system(np.array([x[k,0], x[k,1]])))
                #print(type(obj_val))
                #print(obj_val)
                obj_list.append(obj_val)
                #energy, dev = get_ApartTherm_kpis(x[k, 0], x[k, 1])
                #energy_list.append(energy)
                #dev_list.append(dev)
            obj_arr = np.array(obj_list) #(np.array(energy_list) - energy_mean) / energy_std
            obj_arr = (obj_arr - obj_mean) / obj_std
            #print(obj_arr)
            return obj_arr

        def g_1(x):
            size_batch, _ = x.shape
            constr_list = []
            for k in range(size_batch):
                constr_1 = float(cons_system[0](np.array([x[k,0], x[k,1]])))
                constr_list.append(constr_1)
            constr_arr = np.array(constr_list)/constr_1_std
            #print(constr_arr)
            return constr_arr

        def g_2(x):
            size_batch, _ = x.shape
            constr_list = []
            for k in range(size_batch):
                constr_2 = float(cons_system[1](np.array([x[k,0], x[k,1]])))
                constr_list.append(constr_2)
            constr_arr = np.array(constr_list)/constr_2_std
            #print(constr_arr)
            return constr_arr

        g1_data = g_1(parameter_set)
        g2_data = g_2(parameter_set)
        g1_min = np.min(g1_data)
        g2_min = np.min(g2_data)
        feasible_thr = 0.1
        feasible_points = parameter_set[
            (g1_data <= feasible_thr * g1_min)\
            *(g2_data <= feasible_thr * g2_min),  :]
        feas_points_num, var_dim = feasible_points.shape

        func_feasible_min = np.min(
            f(parameter_set)[
                np.logical_and(g1_data<=0, g2_data<=0)
            ])
        config['f_min'] = func_feasible_min

        config['obj'] = f
        config['constrs_list'] = [g_1, g_2]
        config['vio_cost_funcs_list'] = [cost_funcs['square'], cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square'], cost_funcs_inv['square']]
        config['feasible_points'] = feasible_points
        safe_id = np.random.choice(np.arange(feas_points_num))
        config['init_safe_points'] = np.expand_dims(feasible_points[safe_id],
                                                    axis=0)  # x0

        parameter_num, _ = parameter_set.shape
        random_point_id = np.random.choice(
            np.arange(parameter_num)
        )
        config['init_points'] = np.expand_dims(parameter_set[random_point_id],
                                               axis=0
                                               )
        config['init_safe'] = False #True
        if not config['init_safe']:
            config['init_safe_points'] = config['init_points']
        config['kernel'] = [kernel, constr_kernel_1, constr_kernel_2]
        active_thr = 1e-3
        g_1_active_points = parameter_set[
            np.abs(g_1(parameter_set)) <= active_thr]
        g_2_active_points = parameter_set[
            np.abs(g_2(parameter_set)) <= active_thr]
        config['active_points'] = [g_1_active_points, g_2_active_points]
        print(config)


    if problem_name == 'GP_sample_single_func_2d':
        if problem_dim is None:
            problem_dim = 2
        if gp_kernel is None:
            gp_kernel = 'Gaussian'
        config['var_dim'] = problem_dim
        config['discretize_num_list'] = [30] * problem_dim
        config['num_constrs'] = 1
        config['bounds'] = [(-5, 5)] * problem_dim
        config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            config['discretize_num_list']
        )

        # Measurement noise
        noise_var = 0.05 ** 2

        # Bounds on the inputs variable
        parameter_set = \
            safeopt.linearly_spaced_combinations(config['bounds'],
                                                 config['discretize_num_list'])

        # Define Kernel
        kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=2., \
                              lengthscale=1.0, ARD=True)

        # Initial safe point
        x0 = np.zeros((1, len(config['bounds'])))

        # Generate function with safe initial point at x=0
        def sample_safe_fun():
            while True:
                fun = safeopt.sample_gp_function(kernel, config['bounds'],
                                                 noise_var, 100)
                if fun(0, noise=False) < -0.2:
                    break
            return fun

        func = sample_safe_fun()
        func_min = np.min(func(parameter_set))
        config['f_min'] = func_min
        f = lambda x: func(x).squeeze(axis=1)
        g_1 = lambda x: func(x).squeeze(axis=1)

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['vio_cost_funcs_list'] = [cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]
        config['init_safe_points'] = x0
        config['kernel'] = [kernel, kernel.copy()]

    return config

if __name__ == '__main__':
    a = get_config('GP_sample_two_funcs')
    print(a)
