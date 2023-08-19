import numpy as np
import GPy
from len_scales import get_controller_vars_len_scales, get_twod_gp_len_scales
import safeopt
import sys
sys.path.append('./CSTR')
from sub_uts.systems import *
from sub_uts.utilities_2 import *

ENERGY_THR = 15


def get_config(problem_name, problem_dim=None, gp_kernel=None,
               init_points_id=0, thr_1=0.12, thr_2=0.08,
               vars_to_fix=[], contextual_vars=['p1', 'p2', 'p3', 'p4']):
    if problem_name == 'WO':
        return get_WO_config(problem_name, problem_dim=None, gp_kernel=None,
               init_points_id=0, thr_1=0.12, thr_2=0.08,
               vars_to_fix=[], contextual_vars=['p1', 'p2', 'p3', 'p4'])
    elif problem_name == 'sample_GP_two_dim':
        return get_sample_GP_two_dim_config(problem_name)


def get_sample_GP_two_dim_config(problem_name):
    """
    Input: problem_name
    Output: configuration of the constrained problem, including variable
    dimension, number of constraints, objective function and constraint
    function.
    """
    config = dict()
    config['problem_name'] = 'sample_GP_two_dim'
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
    problem_dim = 2

    # modify this function
    all_vars, all_vars_bounds_dict, all_vars_safe_dict, \
        all_vars_obj_func_len_scales, all_vars_constraint_func_len_scales, \
        all_vars_default_vals = get_twod_gp_len_scales(problem_name)
    vars_to_fix = []

    vars_to_tune = [var for var in all_vars if var not in vars_to_fix]
    tune_var_dim = len(vars_to_tune)
    contextual_vars = ['z']
    contextual_var_ids = [i for i in range(tune_var_dim)
                          if vars_to_tune[i] in contextual_vars]

    # get the ids of contextual variables in tune vars
    config['contextual_var_ids'] = contextual_var_ids
    config['var_dim'] = tune_var_dim
    var_to_optimize_discretize_num = 50
    discrete_num_list = []
    for i in range(tune_var_dim):
        if i in contextual_var_ids:
            discrete_num_list.append(1)
        else:
            discrete_num_list.append(var_to_optimize_discretize_num)

    config['discretize_num_list'] = discrete_num_list
    config['num_constrs'] = 1
    config['bounds'] = [all_vars_bounds_dict[var] for var in vars_to_tune]

    gp_kernel = 'Gaussian'
    noise_var = 0.01
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

    def sample_safe_fun(kernel, config, noise_var, gp_kernel, safe_margin):
        while True:
            print(kernel, noise_var)
            fun = safeopt.sample_gp_function(kernel, config['bounds'],
                                         noise_var, 10)
            # check the uniform Slater's condition

            # go through a net of the original context candidate set
            lw_bound, up_bound = config['bounds'][1]
            is_uniform_Slater = True
            for k in range(var_to_optimize_discretize_num):
                context_num = k * 1.0 / var_to_optimize_discretize_num
                context =  context_num * (up_bound - lw_bound) + lw_bound
                theta_lw, theta_up = config['bounds'][0]
                N = var_to_optimize_discretize_num
                cond_vals = [fun(
                    np.array([m * 1.0/N * (theta_up-theta_lw)+theta_lw,
                              context]), noise=False) for m in range(N)]
                if min(cond_vals) > -safe_margin:
                    is_uniform_Slater = False
            if is_uniform_Slater and \
                    fun(np.array([0, 0]), noise=False) < -safe_margin:
                break

        return fun
    print('Start sampling functions.')
    constr_func = sample_safe_fun(kernel, config, noise_var, gp_kernel, \
                                  safe_margin=0.2)
    obj_func = safeopt.sample_gp_function(kernel, config['bounds'],
                                          noise_var, 10)
    print('Functions sampled.')
    f = lambda x: np.atleast_2d(obj_func(x, noise=False).squeeze(axis=1))
    g_1 = lambda x: constr_func(x, noise=False).squeeze(axis=1)

    config['obj'] = f
    config['constrs_list'] = [g_1]

    lw_bound = -10
    up_bound = 10.0
    context_sequence = []

    for _ in range(1000):
        rand_num = int(np.random.rand() * var_to_optimize_discretize_num) /\
            var_to_optimize_discretize_num
        context =  rand_num * (up_bound - lw_bound) + lw_bound
        context_sequence.append(context)
    context_sequence_len = len(context_sequence)

    def get_context(step=None):
        if step is None:
            step = np.random.randint(context_sequence_len)
        context = context_sequence[step]
        conditional_inputs = [
            [k * 1.0 / var_to_optimize_discretize_num * (up_bound - lw_bound)\
              + lw_bound,
              context ] for k in range(var_to_optimize_discretize_num)
             ]
        # print(conditional_inputs)
        obj_arr = np.array(
            [f(cond_input) for cond_input in conditional_inputs])
        constr_arr = np.array(
            [g_1(cond_input) for cond_input in conditional_inputs])
        cond_min = np.min(obj_arr[constr_arr<=0])
        return context, cond_min

    config['eval_simu'] = False
    config['vio_cost_funcs_list'] = [cost_funcs['linear'],
                                     cost_funcs['linear']]
    config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['linear'],
                                         cost_funcs_inv['linear']]

    safe_point = [all_vars_safe_dict[var] for var in vars_to_tune]
    print(safe_point)
    config['init_safe_points'] = np.array([safe_point])
    config['train_X'] = config['init_safe_points']
    config['get_context'] = get_context
    print(config['var_dim'])
    print(discrete_num_list)
    return config


def get_WO_config(problem_name, problem_dim=None, gp_kernel=None,
               init_points_id=0, thr_1=0.12, thr_2=0.08,
               vars_to_fix=[], contextual_vars=['p1', 'p2', 'p3', 'p4']):
    """
    Input: problem_name
    Output: configuration of the constrained problem, including variable
    dimension, number of constraints, objective function and constraint
    function.
    """
    config = dict()
    config['problem_name'] = problem_name
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


    # modify this function
    all_vars, all_vars_bounds_dict, all_vars_safe_dict, \
        all_vars_obj_func_len_scales, all_vars_constraint1_func_len_scales, \
        all_vars_constraint2_func_len_scales, all_vars_default_vals = \
        get_controller_vars_len_scales(problem_name)
    vars_to_fix = []

    vars_to_tune = [var for var in all_vars if var not in vars_to_fix]
    tune_var_dim = len(vars_to_tune)
    contextual_var_ids = [i for i in range(tune_var_dim)
                          if vars_to_tune[i] in contextual_vars]

    # get the ids of contextual variables in tune vars
    config['contextual_var_ids'] = contextual_var_ids
    config['var_dim'] = tune_var_dim
    var_to_optimize_discretize_num = 50
    discrete_num_list = []
    for i in range(tune_var_dim):
        if i in contextual_var_ids:
            discrete_num_list.append(1)
        else:
            discrete_num_list.append(var_to_optimize_discretize_num)

    config['discretize_num_list'] = discrete_num_list
    config['num_constrs'] = 2
    config['bounds'] = [all_vars_bounds_dict[var] for var in vars_to_tune]

    constr_1_std = 0.03 * 2
    constr_2_std = 4e-4 * 2 * 10

    model = WO_model()
    plant = WO_system()

    obj_model = model.WO_obj_ca
    cons_model = [model.WO_con1_model_ca, model.WO_con2_model_ca]
    obj_system = plant.WO_obj_sys_ca
    obj_system_context = plant.WO_obj_sys_ca_context
    cons_system = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]

    def f(x):
        obj_mean = -52.76428219620413
        obj_std = 22.763438402688305
        size_batch, _ = x.shape
        obj_list = []
        for k in range(size_batch):
            obj_val = float(obj_system_context(np.array([x[k,0], x[k,1],
                                                         x[k, 2], x[k, 3],
                                                         x[k, 4], x[k, 5]])))
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

    context_sequence = []
    lw_ratio = 0.8
    up_ratio = 1.2
    p_base = [1043.38, 20.92, 79.23, 118.34]

    for _ in range(1000):
        context = []
        for k in range(4):
            rand_coef = np.random.rand() * (up_ratio - lw_ratio) + lw_ratio
            context.append(p_base[k] * rand_coef)
        context_sequence.append(context)

    context_sequence_len = len(context_sequence)

    def get_context(step=None):
        if step is None:
            step = np.random.randint(context_sequence_len)
        context = context_sequence[step]
        return context

    config['eval_simu'] = False
    config['obj'] = f
    config['constrs_list'] = [g_1, g_2]
    config['vio_cost_funcs_list'] = [cost_funcs['linear'],
                                     cost_funcs['linear']]
    config['vio_cost_funcs_inv_list'] = [cost_funcs_inv['linear'],
                                         cost_funcs_inv['linear']]

    safe_point = [all_vars_safe_dict[var] for var in vars_to_tune]
    print(safe_point)
    config['init_safe_points'] = np.array([safe_point])
    config['train_X'] = config['init_safe_points']
    config['get_context'] = get_context
    print(config['var_dim'])
    print(discrete_num_list)
    return config

if __name__ == '__main__':
    a = get_config('GP_sample_two_funcs')
    print(a)
