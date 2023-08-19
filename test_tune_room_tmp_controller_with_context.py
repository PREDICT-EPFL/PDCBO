import copy
import numpy as np
import os
import datetime
from tune_util import get_vacbo_optimizer

# parameter configurations to enumerate
discomfort_thr_list = list(range(3, 40, 5))
discomfort_weight_list = list(10 ** np.arange(-3, 2, 1.0))
vabo_budgets_list = [0.001, 0.01, 0.05,
                     0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0]

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
optimization_config = {
    'eval_budget': 100
}

optimizer_base_config = {
    'noise_level': [0.0004, 0.08, 0.2],
    'kernel_var': 0.1,
    'train_noise_level': 1.0,
    'problem_name': 'SinglePIRoomEvaluator',
    'normalize_input': False
}

VARS_TO_FIX = ['high_off_time', 'low_setpoint',
               'control_setpoint']
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

    def evaluate_one_optimizer(
        self, opt_config, optimizer_type,
        discomfort_weights_to_eval=discomfort_weight_list
    ):
        opt_result_dict = dict()

        obj_list_dict = dict()
        constraints_list_dict = dict()
        energy_list_dict = dict()
        discomfort_list_dict = dict()

        evaluated_points_list_dict = dict()
        for discomfort_thr in discomfort_thr_list:
            for discomfort_weight in discomfort_weights_to_eval:
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
                    y_obj, constr_vals = opt.make_step(context_vars)
                    if optimizer_type == 'safe_bo':
                        new_cumu_cost = opt.safe_bo.cumu_vio_cost
                    if optimizer_type == 'constrained_bo':
                        new_cumu_cost = opt.constrained_bo.cumu_vio_cost
                    if optimizer_type == 'violation_aware_bo':
                        new_cumu_cost = opt.violation_aware_bo.cumu_vio_cost
                    if optimizer_type == 'pdcbo':
                        new_cumu_cost = opt.pdbo.cumu_vio_cost
                    if optimizer_type == 'no opt':
                        new_cumu_cost = opt.cumu_vio_cost
                    if optimizer_type == 'grid search':
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
now_time_str = datetime.datetime.now().strftime(
        "%H_%M_%S-%b_%d_%Y")

save_name_append = f'_{tune_var_scale}_with_context_' + \
    f'{optimization_config["eval_budget"]}_' + now_time_str

pdcbo_config = copy.deepcopy(optimizer_base_config)
pdcbo_config.update({
    'kernel_type': 'Gaussian',
    'total_eval_num': optimization_config['eval_budget'],
    'eta_0': 10.0,
    'eta_func': lambda t: 3.0,
    'init_dual': 10.0,
    'lcb_coef': lambda t: 1.0
})
pdcbo_evaluator = OptimizerEvaluator()
pdcbo_evaluator.evaluate_one_optimizer(pdcbo_config,
                                       'pdcbo',
                                       discomfort_weights_to_eval=[0])
pdcbo_evaluator.save_result(
    f'./result/pdcbo{save_name_append}')

for budget in vabo_budgets_list:
    violation_aware_bo_config = copy.deepcopy(optimizer_base_config)
    violation_aware_bo_config.update({
        'single_max_budget': budget,
        'total_vio_budgets': np.array([budget, budget]),
        'prob_eps': 5e-2,
        'beta_0': 1,
        'total_eval_num': optimization_config['eval_budget'],
    })
    violation_aware_bo_evaluator = OptimizerEvaluator()
    violation_aware_bo_evaluator.evaluate_one_optimizer(
        violation_aware_bo_config,
        'violation_aware_bo'
    )
    violation_aware_bo_evaluator.save_result(
        f'./result/violation_aware_bo{save_name_append}_{budget}')

safe_bo_config = copy.deepcopy(optimizer_base_config)
safe_bo_evalutor = OptimizerEvaluator()
safe_bo_evalutor.evaluate_one_optimizer(safe_bo_config, 'safe_bo')
safe_bo_evalutor.save_result(f'./result/safe_bo{save_name_append}')

grid_search_config = copy.deepcopy(optimizer_base_config)
grid_search_config.update({
    'kernel_type': 'Gaussian',
 })
grid_search_evaluator = OptimizerEvaluator()
grid_search_evaluator.evaluate_one_optimizer(grid_search_config,
                                             'grid search'
                                             )
grid_search_evaluator.save_result(
    f'./result/grid_search{save_name_append}')

constrained_bo_config = copy.deepcopy(optimizer_base_config)
constrained_bo_config.update({
    'kernel_type': 'Gaussian',
})
constrained_bo_evaluator = OptimizerEvaluator()
constrained_bo_evaluator.evaluate_one_optimizer(constrained_bo_config,
                                                'constrained_bo')
constrained_bo_evaluator.save_result(
    f'./result/constrained_bo{save_name_append}')
