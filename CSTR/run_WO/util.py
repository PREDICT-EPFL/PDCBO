import numpy as np
import math
import os
import GPy
import safeopt
import merl_model.steady_state_analyze as merl_model
import scipy as sp
import datetime as dt
from run_ApartTherm import get_ApartTherm_kpis

"""
Define some utility functions for the test of safe Bayesian optimization,
constrained Bayesian optimization, and our method.
"""

def try_sample_gp_func(random_sample, num_knots, problem_dim, bounds, config,
                       kernel, numerical_epsilon, noise_var):
    if random_sample:
        sample_list = []
        for i in range(num_knots):
            loc = np.random.uniform(0.0, 1.0, problem_dim)
            x = np.array([bounds[i][0] +
                         (bounds[i][1] - bounds[i][0]) * loc[i]
                          for i in range(problem_dim)]
                         )
            sample_list.append(x)
        knot_x = np.array(sample_list)
    else:
        knot_x = safeopt.linearly_spaced_combinations(
            config['bounds'],
            config['discretize_num_list']
        )

    config['knot_x'] = knot_x
    knot_cov = kernel.K(config['knot_x']) + \
        np.eye(knot_x.shape[0]) * numerical_epsilon
    knot_cov_cho = sp.linalg.cho_factor(knot_cov)
    fun = safeopt.sample_gp_function(kernel,
                                     config['bounds'],
                                     noise_var,
                                     config['discretize_num_list']
                                     )
    knot_y = fun(knot_x)
    alpha = sp.linalg.cho_solve(knot_cov_cho, knot_y)

    def obj_f(x):
        x = np.atleast_2d(x)
        y = kernel.K(x, knot_x).dot(alpha)
        y = np.squeeze(y)
        return y

    #config['knot_cov'] = knot_cov
    #config['knot_cov_cho'] = knot_cov_cho
    #config['knot_y'] = knot_y
    #config['knot_min'] = np.min(knot_y)
    #config['alpha'] = alpha
    func_norm = (knot_y.T @ alpha)[0, 0]
    return obj_f, func_norm


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

    if problem_name == '2d_dim_merl_refri':
        config['var_dim'] = 2
        config['discretize_num_list'] = [21, 48]
        config['num_constrs'] = 1
        config['bounds'] = [(140, 340), (400, 870)]
        config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            config['discretize_num_list']
        )

        data = np.load('./data/res_10_dim_2_fix_200.npz', allow_pickle=True)
        powers = data['arr_0']
        TD_is = data['arr_1']
        TD_is_cons = 273.15 + 100
        f = lambda x: powers[((x[:, 0]-100)/10.0).astype('int'), 0, \
                             ((x[:, 1]-400)/10.0).astype('int')]
        g_1 = lambda x: TD_is[((x[:, 0]-100)/10.0).astype('int'), 0,\
                              ((x[:, 1]-400)/10.0).astype('int')] \
            - TD_is_cons

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['vio_cost_funcs_list'] = [cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]
        config['init_safe_points'] = np.array([[230.0, 610.0]
                                               #[240.0, 600.0],
                                               #[240.0, 610.0]
                                               ]
                                              )

    if problem_name == 'merl_refri_fix_u1':
        if gp_kernel is None:
            gp_kernel = 'Gaussian'

        config['var_dim'] = 2
        data = np.load('./data/res_10_dim_1_fix_300.npz', allow_pickle=True)
        powers = data['power']
        TD_is = data['TDis']
        u1 = data['u1']
        u2 = data['u2']

        config['discretize_num_list'] = [len(u1), len(u2)]
        config['num_constrs'] = 1
        config['bounds'] = [(u1[0], u1[-1]), (u2[0], u2[-1])]
        config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            config['discretize_num_list']
        )
        range_power_u1 = np.max(powers, axis=0) - np.min(powers, axis=0)
        range_power = np.max(powers) - np.min(powers)
        range_TDis = np.max(TD_is) - np.min(TD_is)
        u1_range = u1[-1] - u1[0]
        u2_range = u2[-1] - u2[0]
        diam_u = math.sqrt((u1[-1]-u1[0])**2 + (u2[-1]-u2[0])**2)

        if gp_kernel == 'Gaussian':
            kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=(range_power/10.0)**2,
                                  lengthscale=[u1_range/10, u2_range/10], ARD=True)
            constr_kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=(range_TDis/10.0)**2,
                                  lengthscale=[u1_range/10, u2_range/10], ARD=True)

        TD_is_cons = 273.15 + 60
        f = lambda x: powers[((x[:, 0]-u1[0])/10.0).astype('int'),
                             ((x[:, 1]-u2[0])/10.0).astype('int')]
        g_1 = lambda x: TD_is[((x[:, 0]-u1[0])/10.0).astype('int'),
                              ((x[:, 1]-u2[0])/10.0).astype('int')] \
            - TD_is_cons

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['vio_cost_funcs_list'] = [cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]
        config['init_safe_points'] = np.array([[u1[-3], u2[-3]],
                                               [u1[-2], u2[-2]],
                                                [u1[-2], u2[-3]]
                                               #[240.0, 600.0],
                                               #[240.0, 610.0]
                                              ]
                                              )

        config['kernel'] = [kernel, constr_kernel]

    if problem_name == 'merl_simulator':
        if gp_kernel is None:
            gp_kernel = 'Gaussian'

        config['var_dim'] = 5

        config['discretize_num_list'] = [60, 60, 100, 1, 1]
        config['num_constrs'] = 1
        config['bounds'] = [(200, 350), (300, 450), (500, 850), (0.6, 0.8),
                            (28, 32)]

        if gp_kernel == 'Gaussian':
            kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=15.0,
                                  lengthscale=[50, 60, 70, 50, 0.3], ARD=True)
            constr_kernel = GPy.kern.RBF(input_dim=len(config['bounds']),
                                         variance=5.0,  #2.0,
                                         lengthscale=[20, 24, 28, 50, 0.3],
                                         ARD=True)

        power_shift = 500
        TD_is_cons = 273.15 + 60 #+ 58
        def f(x, simulator_to_use=None):
            size_batch, _ = x.shape
            power_list = []
            TD_is_list = []
            for k in range(size_batch):
                steady_power, steady_TDis, simulator_to_use = \
                    merl_model.evaluate_one_point([x[k, 0], x[k, 1], x[k, 2]],
                                                  simulator_to_use=simulator_to_use)
                power_list.append(steady_power)
                TD_is_list.append(steady_TDis)
            power_arr = np.array(power_list) - power_shift
            TD_is_arr = np.array(TD_is_list) - TD_is_cons
            return power_arr, simulator_to_use, TD_is_arr

        def g_1(x, simulator_to_use=None):
            size_batch, _ = x.shape
            TD_is_list = []
            power_list = []
            for k in range(size_batch):
                steady_power, steady_TDis, simulator_to_use = merl_model.evaluate_one_point([x[k, 0], x[k, 1], x[k, 2]],
                                                                                            simulator_to_use=simulator_to_use)
                TD_is_list.append(steady_TDis)
                power_list.append(steady_power)
            power_arr = np.array(power_list) - power_shift
            TD_is_arr = np.array(TD_is_list) - TD_is_cons
            return TD_is_arr, simulator_to_use, power_arr

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['vio_cost_funcs_list'] = [cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]
        config['init_safe_points'] = np.array([
            [300, 430, 840, 30, 0.6],
            [320, 420, 800, 30, 0.6],
            [310, 390, 810, 30, 0.6]
                                               ]
                                              )
        config['train_X'] = config['init_safe_points']
        config['kernel'] = [kernel, constr_kernel]
        config['contextual_var_ids'] = [3, 4]
        config['single_max_budget'] = 40 ** 2

    if problem_name == 'merl_simulator_with_TEvap':
        if gp_kernel is None:
            gp_kernel = 'Gaussian'

        config['var_dim'] = 3

        config['discretize_num_list'] = [60, 60, 100]
        config['num_constrs'] = 2
        config['bounds'] = [(200, 350), (300, 450), (500, 850)]
        #[(200, 300), (300, 400), (500, 800)]
        #[(200, 350), (300, 450), (500, 850)]
        safe_ranges = [(260, 300), (360, 400), (700, 800)]
        if gp_kernel == 'Gaussian':
            kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=15.0,
                                  lengthscale=[50, 60, 70], ARD=True)
            constr_kernel_1 = GPy.kern.RBF(input_dim=len(config['bounds']), variance=2.0,
                                           lengthscale=[20, 24, 28], ARD=True)
            constr_kernel_2 = GPy.kern.RBF(input_dim=len(config['bounds']), variance=2.0,
                                           lengthscale=[20, 24, 28], ARD=True)

        power_shift = 500
        TD_is_cons = 273.15 + 58
        TEvap_cons = 273.15 + 5

        def f(x, simulator_to_use=None):
            size_batch, _ = x.shape
            power_list = []
            TD_is_list = []
            TEvap_list = []
            for k in range(size_batch):
                steady_power, steady_TDis, TEvap, simulator_to_use = merl_model.evaluate_one_point_with_TEvap(
                    [x[k, 0], x[k, 1], x[k, 2]],
                    simulator_to_use=simulator_to_use
                )
                power_list.append(steady_power)
                TD_is_list.append(steady_TDis)
                TEvap_list.append(TEvap)

            power_arr = np.array(power_list) - power_shift
            TD_is_arr = np.array(TD_is_list) - TD_is_cons
            TEvap_arr = TEvap_cons - np.array(TEvap_list)
            return power_arr, simulator_to_use, TD_is_arr, TEvap_arr

        def g_1(x, simulator_to_use=None):
            size_batch, _ = x.shape
            TD_is_list = []
            power_list = []
            TEvap_list = []
            for k in range(size_batch):
                steady_power, steady_TDis, TEvap, simulator_to_use = merl_model.evaluate_one_point_with_TEvap(
                    [x[k, 0], x[k, 1], x[k, 2]],
                    simulator_to_use=simulator_to_use
                )
                TD_is_list.append(steady_TDis)
                power_list.append(steady_power)
                TEvap_list.append(TEvap)

            power_arr = np.array(power_list) - power_shift
            TD_is_arr = np.array(TD_is_list) - TD_is_cons
            TEvap_arr = TEvap_cons - np.array(TEvap_list)
            return TD_is_arr, simulator_to_use, TEvap_arr, power_arr

        def g_2(x, simulator_to_use=None):
            size_batch, _ = x.shape
            TD_is_list = []
            power_list = []
            TEvap_list = []
            for k in range(size_batch):
                steady_power, steady_TDis, TEvap, simulator_to_use = merl_model.evaluate_one_point_with_TEvap(
                    [x[k, 0], x[k, 1], x[k, 2]],
                    simulator_to_use=simulator_to_use)
                TD_is_list.append(steady_TDis)
                power_list.append(steady_power)
                TEvap_list.append(TEvap)

            power_arr = np.array(power_list) - power_shift
            TD_is_arr = np.array(TD_is_list) - TD_is_cons
            TEvap_arr = TEvap_cons - np.array(TEvap_list)
            return TEvap_arr, simulator_to_use, power_arr, TD_is_arr

        config['obj'] = f
        config['constrs_list'] = [g_1, g_2]
        config['vio_cost_funcs_list'] = [cost_funcs['square'], cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square'], cost_funcs_inv['square']]
        if init_points_id == 0:
            config['init_safe_points'] = np.array([
                [300, 430, 840],
                [320, 420, 800],
                [310, 390, 810]
            ]
            )
        elif init_points_id == 1:
            config['init_safe_points'] = np.array([
                [300, 430, 840]
            ]
            )
        elif init_points_id == 2:
            config['init_safe_points'] = np.array([
                [320, 420, 800]
            ]
            )
        elif init_points_id == 3:
            config['init_safe_points'] = np.array([
                [320, 420, 800]
            ]
            )
        elif init_points_id == 4:
            config['init_safe_points'] = np.array([
                [290, 400, 780]
            ]
            )
        elif init_points_id == 'random':
            init_safe_points_list = []
            for k in range(3):
                safe_point = []
                for l in range(3):
                    x = (safe_ranges[l][1]-safe_ranges[l][0]) * np.random.rand() + safe_ranges[l][0]
                    safe_point.append(x)
                init_safe_points_list.append(safe_point)
            config['init_safe_points'] = np.array(init_safe_points_list
            )
        elif 'random_id_' in init_points_id:
            _, _, random_id = init_points_id.strip().split('_')
            file_init_points = './data/init_points_data/'+init_points_id
            if os.path.exists(file_init_points+'.npz'):
                config['init_safe_points'] = np.load(file_init_points+'.npz', allow_pickle=True)['arr_0'][0]
            else:
                init_safe_points_list = []
                for k in range(3):
                    safe_point = []
                    for l in range(3):
                        x = (safe_ranges[l][1] - safe_ranges[l][0]) * np.random.rand() + safe_ranges[l][0]
                        safe_point.append(x)
                    init_safe_points_list.append(safe_point)
                config['init_safe_points'] = np.array(init_safe_points_list
                                                      )
                np.savez(file_init_points, [config['init_safe_points']])

        config['train_X'] = config['init_safe_points']
        config['kernel'] = [kernel, constr_kernel_1, constr_kernel_2]

    if problem_name == '1d_dim_merl_refri':
        config['var_dim'] = 1
        config['discretize_num_list'] = [50]
        config['num_constrs'] = 1
        config['bounds'] = [(400, 890)]
        config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            config['discretize_num_list']
        )
        if gp_kernel == 'Gaussian':
            kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=(30.0 / 4.0) ** 2,
                                  lengthscale=[500/10.0], ARD=True)
            constr_kernel = GPy.kern.RBF(input_dim=len(config['bounds']), variance=( 10.0 / 4.0) ** 2,
                                         lengthscale=[500/3.0], ARD=True)

        config['kernel'] = [kernel, constr_kernel]

        data = np.load('./data/u1_300_u2_300_tune_u3.npz', allow_pickle=True)
        powers = data['arr_0']
        TD_is = data['arr_1']
        TD_is_cons = 273.15 + 60
        f = lambda x: powers[((x[:, 0]-400)/10.0).astype('int')] #- 425
        g_1 = lambda x: TD_is[((x[:, 0]-400)/10.0).astype('int')] - TD_is_cons

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['vio_cost_funcs_list'] = [cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]
        config['init_safe_points'] = np.array([[880],
                                               [870]
                                               ]
                                              )


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
        func = sample_safe_fun(
            kernel, config, noise_var, gp_kernel, safe_margin, num_bound=10)
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
        config['eta_func'] = lambda t: 3/np.sqrt(t * 1.0)
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
        # Generate function with safe initial point at x=0
        def sample_safe_fun(kernel, config, noise_var, gp_kernel, safe_margin,
                            norm_bound, num_knots=3):
            while True:
                fun, fun_norm = try_sample_gp_func(
                    random_sample=True, num_knots=num_knots,
                    problem_dim=config['var_dim'], bounds=config['bounds'],
                    config=config, kernel=gp_kernel, numerical_epsilon=1e-10,
                    noise_var=1e-5)
                if gp_kernel == 'Gaussian':
                    if fun(0, noise=False) < -safe_margin \
                            and fun_norm <= norm_bound:
                        break
                if gp_kernel == 'poly':
                    if fun(0, noise=False) < -safe_margin \
                           and fun_norm <= norm_bound:
                        break
            return fun

        def sample_bounded_norm_fun(
            kernel, config, noise_var, gp_kernel, safe_margin,
            norm_bound, num_knots=3):
            while True:
                fun, fun_norm = try_sample_gp_func(
                    random_sample=True, num_knots=num_knots,
                    problem_dim=config['var_dim'], bounds=config['bounds'],
                    config=config, kernel=gp_kernel, numerical_epsilon=1e-10,
                    noise_var=1e-5)
                if gp_kernel == 'Gaussian':
                    if fun_norm <= norm_bound:
                        break
                if gp_kernel == 'poly':
                    if fun_norm <= norm_bound:
                        break
            return fun


        constr_func = sample_safe_fun(kernel, config, noise_var, gp_kernel,
                                      safe_margin=0.2, norm_bound=1.0)
        obj_func = sample_bounded_norm_fun(kernel, config, noise_var, gp_kernel,
                                      safe_margin=0.2, norm_bound=1.0)
        func_feasible_min = np.min(
            obj_func(parameter_set)[constr_func(parameter_set)<=0])
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
        config['bounds'] = [(-10, 10)] * problem_dim
        config['train_X'] = safeopt.linearly_spaced_combinations(
            config['bounds'],
            config['discretize_num_list']
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
                lengthscale=[0.3, 1.0], ARD=True)

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
            for k in range(size_batch):
                energy, dev = get_ApartTherm_kpis(x[k, 0], x[k, 1])
                energy_list.append(energy)
                dev_list.append(dev)
            energy_arr = (np.array(energy_list) - energy_mean) / energy_std
            dev_arr = np.array(TD_is_list) - TD_is_cons
            return power_arr, simulator_to_use, TD_is_arr

        def g_1(x, simulator_to_use=None):
            size_batch, _ = x.shape
            TD_is_list = []
            power_list = []
            for k in range(size_batch):
                steady_power, steady_TDis, simulator_to_use = merl_model.evaluate_one_point([x[k, 0], x[k, 1], x[k, 2]],
                                                                                            simulator_to_use=simulator_to_use)
                TD_is_list.append(steady_TDis)
                power_list.append(steady_power)
            power_arr = np.array(power_list) - power_shift
            TD_is_arr = np.array(TD_is_list) - TD_is_cons
            return TD_is_arr, simulator_to_use, power_arr

        config['obj'] = f
        config['constrs_list'] = [g_1]
        config['vio_cost_funcs_list'] = [cost_funcs['square']]
        config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['square']]
        config['init_safe_points'] = x0
        config['kernel'] = [kernel, constr_kernel]


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
