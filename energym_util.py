import numpy as np
from EMPA_VABO_experiments import pcnn_room_v2, rc_room, \
    controlled_room
from EMPA_VABO_experiments.room import LinearRoomModel
from copy import deepcopy
import GPy
import datetime as dt
from EMPA_VABO_experiments import room_controller_evaluator
from util import get_dict_medium


"""
Define some utility functions for the test of safe Bayesian optimization,
constrained Bayesian optimization, and our method for room temperature
controller tuning.
"""


def get_energym_room_controller_vars_len_scales():
    """
    6 vars to tune:
    lower_tol,
    upper_tol,
    nighttime_setback,
    nighttime_start,
    nighttime_end,
    nighttime_temp
    """
    all_vars = ['lower_tol', 'upper_tol',
                'nighttime_start', 'nighttime_end'
                ]

    all_vars_bounds_dict = {
                           }
    all_vars_safe_dict = {
                          }
    all_vars_energy_func_len_scales = {
                                       }
    all_vars_discomfort_func_len_scales = {
    }

    all_vars_default_vals = get_dict_medium(all_vars_bounds_dict)

    return all_vars, all_vars_bounds_dict, all_vars_safe_dict, \
        all_vars_energy_func_len_scales, all_vars_discomfort_func_len_scales, \
        all_vars_default_vals

def get_config(problem_name, problem_dim=None, gp_kernel=None,
               init_points_id=0, discomfort_thr=1.0, max_discomfort_thr=5,
               vars_to_fix=[], start_eval_time=None, room_simulator='Linear',
               contextual_vars=[], discomfort_weight=0.01,
               tune_PI_scale='linear', tune_obj='energy', energy_thr=ENERGY_SHIFT):
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

    # transform the discomfort budget in one day into average temperature
    # deviation on one data point
    discomfort_thr = discomfort_thr / 24.0
    if problem_name == 'Apartments2Thermal':
        all_vars, all_vars_bounds_dict, all_vars_safe_dict, \ 
        all_vars_energy_func_len_scales, all_vars_discomfort_func_len_scales, \
        all_vars_max_discomfort_func_len_scales, all_vars_default_vals = \
            get_tmp_controller_vars_len_scales(problem_name,
                                               tune_PI_scale=tune_PI_scale,
                                               tune_obj=tune_obj)

        vars_to_tune = [var for var in all_vars if var not in vars_to_fix]
        tune_var_dim = len(vars_to_tune)
        contextual_var_ids = [i for i in range(tune_var_dim)
                              if vars_to_tune[i] in contextual_vars]

        # get the ids of contextual variables in tune vars
        config['contextual_var_ids'] = contextual_var_ids
        config['var_dim'] = tune_var_dim
        var_to_optimize_discretize_num = 20
        discrete_num_list = []
        for i in range(tune_var_dim):
            if i in contextual_var_ids:
                discrete_num_list.append(1)
            else:
                discrete_num_list.append(var_to_optimize_discretize_num)
        config['discretize_num_list'] = discrete_num_list
        config['num_constrs'] = 1    # 1 constraint on discomfort
        config['bounds'] = [all_vars_bounds_dict[var] for var in vars_to_tune]
        tune_vars_energy_len_scales = [all_vars_energy_func_len_scales[var]
                                       for var in vars_to_tune]
        tune_vars_discomfort_len_scales = [
            all_vars_discomfort_func_len_scales[var]
            for var in vars_to_tune
        ]
        
        if gp_kernel is None or gp_kernel == 'Gaussian':
            kernel = GPy.kern.RBF(
                input_dim=len(config['bounds']),
                variance=0.001/25.0,
                lengthscale=tune_vars_energy_len_scales,
                ARD=True
            )
            constr_kernel_1 = GPy.kern.RBF(
                input_dim=len(config['bounds']),
                variance=0.05 * 8,
                lengthscale=tune_vars_discomfort_len_scales,
                ARD=True
            )
        elif gp_kernel == 'Matern52':
            kernel = GPy.kern.Matern52(
                input_dim=len(config['bounds']),
                variance=0.001/25.0,
                lengthscale=tune_vars_energy_len_scales,
                ARD=True
            )
            constr_kernel_1 = GPy.kern.Matern52(
                input_dim=len(config['bounds']),
                variance=0.05 * 8,
                lengthscale=tune_vars_discomfort_len_scales,
                ARD=True
            )
        if tune_obj == 'energy':
            config['kernel'] = [kernel, constr_kernel_1, constr_kernel_2]
        elif tune_obj == 'discomfort':
            config['kernel'] = [constr_kernel_1, kernel, constr_kernel_2]
        else:
            print('Unsupported tune obj.')

        global ENERGY_SHIFT
        config['energy_shift'] = energy_thr
        config['discomfort_thr'] = discomfort_thr
        config['max_discomfort_thr'] = max_discomfort_thr

        def f(x, simulator_to_use=None, is_look_ahead=False):
            size_batch, num_tune_vars = x.shape
            energy_list = []
            energy_shift = energy_thr
            discomfort_list = []
            if simulator_to_use is None:
                simulator_to_use = room_controller_evaluator.\
                            SingleRoomEvaluator(
                                PI_controlled_room,
                                temp_discomfort_cost,
                                ambient_file
                            )


            for k in range(size_batch):
                tmp_config = deepcopy(all_vars_default_vals)
                for var_id in range(num_tune_vars):
                    var = vars_to_tune[var_id]
                    tmp_config[var] = x[k, var_id]

                if is_bang_bang_controller:
                    controller_config = {
                        'tmp_thresholds': [tmp_config['open_temp'],
                                           tmp_config['close_temp']],
                        'RB_valve_ratio': tmp_config['input_power'],
                        'RB_start_time': tmp_config['start_time'],
                        'RB_end_time': tmp_config['end_time']
                                           }
                elif is_PI_controller:
                    controller_config = {
                        'high_on_time': tmp_config['high_on_time'],
                        'high_off_time': tmp_config['high_off_time'],
                        'P': tmp_config['P'],
                        'I': tmp_config['I'],
                        'high_setpoint': tmp_config['high_setpoint'],
                        'low_setpoint': tmp_config['low_setpoint'],
                        'control_setpoint': tmp_config['control_setpoint']
                        # 'temperature': tmp_config['low_setpoint']
                                           }
                    if tune_PI_scale == 'log':
                        controller_config['P'] = np.exp(controller_config['P'])
                        controller_config['I'] = np.exp(controller_config['I'])

                simulator_to_use.controlled_room.reset_control_config(
                    controller_config)
                if not is_look_ahead:
                    energy, discomfort, max_discomfort = \
                        simulator_to_use.run_one_episode(epi_len= 96 * 1)
                    #60 * 24)
                else:
                    energy, discomfort, max_discomfort = \
                        simulator_to_use.look_ahead_one_episode()
                    return energy, discomfort, max_discomfort
#96 * 1) #60 * 24)
                energy_list.append(energy)
                discomfort_list.append(discomfort)
                max_discomfort_list.append(max_discomfort)
            energy_arr = np.array(energy_list) - energy_shift
            discomfort_arr = np.array(discomfort_list) - discomfort_thr
            max_discomfort_arr = np.array(max_discomfort_list) - \
                max_discomfort_thr
            # obj_dim: (batch_size), constr_dim: (batch_size, num_of_constr)
            # expand the dimension
            print(energy_list, discomfort_list, max_discomfort_list)
            if tune_obj == 'energy':
                obj_arr = energy_arr  #+ discomfort_weight * discomfort_arr
                constr_arr = np.stack((discomfort_arr, max_discomfort_arr),
                                  axis=-1)
            elif tune_obj == 'discomfort':
                obj_arr = discomfort_arr
                constr_arr = np.stack((energy_arr, max_discomfort_arr),
                                  axis=-1)

            return obj_arr, constr_arr, simulator_to_use

        def get_context(simulator_to_use=None):
            if simulator_to_use is None:
                if is_bang_bang_controller:
                    if room_simulator == 'Linear':
                        simulator_to_use = room_controller_evaluator.\
                            SingleRBRoomEvaluator(
                                start_date_time,
                                STEP_DURATION,
                                LinearRoomParams['T_room'],
                                LinearRoomParams['uk'],
                                LinearRoomParams['T_neighbors'],
                                LinearRoomParams['T_amb'],
                                LinearRoomParams['s_coefs'],
                                MAX_VALVE_POWER,
                                'heating',
                                tmp_thresholds=[22, 24],
                                RB_valve_ratio=1.0,
                                ambient_file=ambient_file,
                                temp_discomfort_cost=temp_discomfort_cost,
                                RB_start_time=6 * 60,
                                RB_end_time=18 * 60
                            )
                    elif room_simulator == 'PCNN':
                        room = pcnn_room_v2.PCNNRoomV2Model(
                            start_date_time,
                            STEP_DURATION,
                            23)
                        # controlled_room = controlled_room.PIControlledRoom(
                    elif room_simulator == 'rc':
                        room = rc_room.RCRoomModel(
                            start_date_time,
                            60,
                            21.5)
                elif is_PI_controller:
                    if room_simulator == 'Linear':
                        linear_room = LinearRoomModel(
                            start_date_time,
                            STEP_DURATION,
                            21,
                            LinearRoomParams['T_room'],
                            LinearRoomParams['uk'],
                            LinearRoomParams['T_neighbors'],
                            LinearRoomParams['T_amb'],
                            LinearRoomParams['s_coefs'],
                        )
                        PI_controlled_room = controlled_room.PIControlledRoom(
                            linear_room,
                            MAX_VALVE_POWER,
                            'heating',
                            high_on_time=0 * 60,
                            high_off_time=19 * 60,
                            P_coef=0.3,
                            I_coef=0.2,
                            high_setpoint=22.5,
                            low_setpoint=19.0,
                            control_setpoint=0.0
                        )
                        simulator_to_use = room_controller_evaluator.\
                            SingleRoomEvaluator(
                                PI_controlled_room,
                                temp_discomfort_cost,
                                ambient_file)

                    elif room_simulator == 'PCNN':
                        room = pcnn_room_v2.PCNNRoomV2Model(
                            start_date_time,
                            STEP_DURATION,
                            21.5)
                        PI_controlled_room = controlled_room.PIControlledRoom(
                            room,
                            MAX_VALVE_POWER,
                            'heating',
                            high_on_time=0 * 60,
                            high_off_time=19 * 60,
                            P_coef=0.3,
                            I_coef=0.2,
                            high_setpoint=22.5,
                            low_setpoint=19.0,
                            control_setpoint=0.0
                        )
                        simulator_to_use = room_controller_evaluator.\
                            SingleRoomEvaluator(
                                PI_controlled_room,
                                temp_discomfort_cost,
                                ambient_file)
                        print('PI controlled PCNN room.')
                    elif room_simulator == 'rc':
                        room = rc_room.RCRoomModel(
                            start_date_time,
                            60,
                            21.5)
                        PI_controlled_room = controlled_room.PIControlledRoom(
                            room,
                            MAX_VALVE_POWER,
                            'heating',
                            high_on_time=0 * 60,
                            high_off_time=19 * 60,
                            P_coef=0.3,
                            I_coef=0.2,
                            high_setpoint=22.5,
                            low_setpoint=19.0,
                            control_setpoint=0.0
                        )
                        simulator_to_use = room_controller_evaluator.\
                            SingleRoomEvaluator(
                                PI_controlled_room,
                                temp_discomfort_cost,
                                ambient_file
                            )
            else:
                room = simulator_to_use.controlled_room.room
            context = room.predict_context(
                24 * 4, ['Q_irr', 'T_out', 'T_init'])
            return context

        config['eval_simu'] = True
        config['obj'] = f
        config['constrs_list'] = []
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
