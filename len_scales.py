import numpy as np
from temp_cost import temp_discomfort_cost, get_dict_medium

def get_twod_gp_len_scales(problem_name):
    """
    2 vars to tune
    """
    all_vars = ['theta', 'z']
    all_vars_bounds_dict = {
        'theta': [-10, 10],
        'z': [-10, 10]
        }
    all_vars_safe_dict = {
        'theta': 0.0,
        'z': 0.0,
    }
    all_vars_obj_func_len_scales = {
        'theta': 3.0,
        'z': 3.0
    }
    all_vars_constraint_func_len_scales = all_vars_obj_func_len_scales

    all_vars_default_vals = get_dict_medium(all_vars_bounds_dict)

    return all_vars, all_vars_bounds_dict, all_vars_safe_dict, \
        all_vars_obj_func_len_scales, all_vars_constraint_func_len_scales, \
        all_vars_default_vals


def get_controller_vars_len_scales(problem_name='WO'):
    """
    2 vars to tune:
    """
    lw_ratio = 0.8
    up_ratio = 1.2
    all_vars = ['fr', 'tmp', 'p1', 'p2', 'p3', 'p4']
    all_vars_bounds_dict = {
        'fr': [4, 7],
        'tmp': [70, 100],
        'p1': [1043.38 * lw_ratio , 1043.38 * up_ratio],
        'p2': [20.92 * lw_ratio , 20.92 * up_ratio],
        'p3': [79.23 * lw_ratio ,  79.23 * up_ratio],
        'p4': [118.34 * lw_ratio ,  118.34 * up_ratio]
        }
    all_vars_safe_dict = {
        'fr': 4.1,
        'tmp': 81.2,
        'p1': 1043.38,
        'p2': 20.92,
        'p3': 79.23,
        'p4': 118.34
    }
    all_vars_obj_func_len_scales = {
        'fr': 1.0,
        'tmp': 10.0,
        'p1': 100.0,
        'p2': 2,
        'p3': 7,
        'p4': 10
    }
    all_vars_constraint1_func_len_scales = all_vars_obj_func_len_scales
    all_vars_constraint2_func_len_scales = all_vars_obj_func_len_scales
    all_vars_default_vals = get_dict_medium(all_vars_bounds_dict)

    return all_vars, all_vars_bounds_dict, all_vars_safe_dict, \
        all_vars_obj_func_len_scales, all_vars_constraint1_func_len_scales, \
        all_vars_constraint2_func_len_scales, all_vars_default_vals
