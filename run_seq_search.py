import copy
import numpy as np
import os
from tune_util import get_vacbo_optimizer

# parameter configurations to enumerate
discomfort_thr_list = [5]  # list(range(5, 5, 10))
discomfort_weight_list = [0.1]
weight_list = 10 ** (np.arange(-4, 3, 1))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
optimization_config = {
    'eval_budget': 40
}

optimizer_base_config = {
    'noise_level': [0.004, 0.2, 0.2],
    'kernel_var': 0.1,
    'train_noise_level': 1.0,
    'problem_name': 'SinglePIRoomEvaluator',
    'normalize_input': False
}

VARS_TO_FIX = ['high_on_time', 'high_off_time', 'high_setpoint',
               'low_setpoint', 'control_setpoint']
CONTEXTUAL_VARS = ['Q_irr', 'T_out', 'T_init']


class OptimizerEvaluator:

    def __init__(self):
        self.opt_result_dict = None
        self.obj_list_dict = None
        self.constraints_list_dict = None
        self.energy_list_dict = None
        self.discomfort_list_dict = None
        self.seasonal_energy_list_dict = None
        self.seasonal_discomfort_list_dict = None
        self.evaluated_points_list_dict = None

    def evaluate_one_optimizer(self, opt_config, optimizer_type):
        opt_result_dict = dict()

        obj_list_dict = dict()
        constraints_list_dict = dict()
        energy_list_dict = dict()
        discomfort_list_dict = dict()

        evaluated_points_list_dict = dict()
        for discomfort_thr in discomfort_thr_list:
            for discomfort_weight in discomfort_weight_list:
                opt, opt_total_cost_list, opt_problem = get_vacbo_optimizer(
                    opt_config['problem_name'], optimizer_type, opt_config,
                    discomfort_thr=discomfort_thr, vars_to_fix=VARS_TO_FIX,
                    contextual_vars=CONTEXTUAL_VARS,
                    discomfort_weight=discomfort_weight)
                opt_obj_list = []
                constraints_list = []
                energy_list = []
                discomfort_list = []
                for _ in range(optimization_config['eval_budget']):
                    context_vars = opt_problem.get_context(
                        opt_problem.simulator)
                    y_obj, constr_vals = opt.make_step()
                    if optimizer_type == 'safe_bo':
                        new_cumu_cost = opt.safe_bo.cumu_vio_cost
                    if optimizer_type == 'constrained_bo':
                        new_cumu_cost = opt.constrained_bo.cumu_vio_cost
                    if optimizer_type == 'violation_aware_bo':
                        new_cumu_cost = opt.violation_aware_bo.cumu_vio_cost
                    if optimizer_type == 'no opt':
                        new_cumu_cost = opt.cumu_vio_cost
                    if optimizer_type == 'grid search' or 'seq grid search':
                        new_cumu_cost = opt.cumu_vio_cost

                    opt_total_cost_list.append(new_cumu_cost)
                    opt_obj_list.append(y_obj)
                    constraints_list.append(constr_vals)
                    energy, discomfort = opt_problem.simulator.\
                        get_recent_energy_discomfort_per_day()

                    energy_list.append(energy)
                    discomfort_list.append(discomfort)
                    print_log = True
                    if print_log:
                        print(f"In step {_}, with discomfort threshold " +
                              f"{discomfort_thr} and discomfort weight " +
                              f"{discomfort_weight}, we get energy {energy}" +
                              f" and discomfort {discomfort}, with the point "
                              + f" {opt_problem.evaluated_points_list[-1]}.")

                opt_config_key = f'({discomfort_thr},{discomfort_weight})'

                opt_result_dict[opt_config_key] = opt
                obj_list_dict[opt_config_key] = opt_obj_list
                constraints_list_dict[opt_config_key] = constraints_list
                energy_list_dict[opt_config_key] = energy_list
                discomfort_list_dict[opt_config_key] = discomfort_list

                evaluated_points_list_dict[opt_config_key] = opt_problem.\
                    evaluated_points_list

        self.opt_result_dict = opt_result_dict
        self.obj_list_dict = obj_list_dict
        self.constraints_list_dict = constraints_list_dict
        self.energy_list_dict = energy_list_dict
        self.discomfort_list_dict = discomfort_list_dict
        self.evaluated_points_list_dict = evaluated_points_list_dict

    def save_result(self, save_path):
        np.savez(save_path, self.obj_list_dict, self.constraints_list_dict,
                 self.energy_list_dict, self.discomfort_list_dict,
                 self.evaluated_points_list_dict)


tune_var_scale = 'log'

for weight in weight_list:
    save_name_append = f'_{weight}_{tune_var_scale}_with_context'
    grid_search_config = copy.deepcopy(optimizer_base_config)
    grid_search_config.update({
        'kernel_type': 'Gaussian',
    })
    grid_search_evaluator = OptimizerEvaluator()
    grid_search_config['discomfort_weight'] = weight
    grid_search_evaluator.evaluate_one_optimizer(grid_search_config,
                                                 'seq grid search'
                                                 )
    grid_search_evaluator.save_result(
        f'./result/try_seq_grid_search{save_name_append}')
