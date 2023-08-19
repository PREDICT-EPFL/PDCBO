import copy
import numpy as np
import util
import matplotlib.pyplot as plt
import os
import vabo
from keep_default_optimizer import KeepDefaultOpt
from grid_search_optimizer import GridSearchOpt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
optimization_config = {
    'eval_budget': 40  # 20
}
discomfort_thr_list = list(range(5, 6))
optimizer_base_config = {
    'noise_level': [0.1, 0.1, 0.2],
    'kernel_var': 0.1,
    'train_noise_level': 1.0,
    'problem_name': 'SinglePIRoomEvaluator',
    'normalize_input': False
}

vars_to_fix = ['high_on_time', 'high_off_time', 'high_setpoint',
               'low_setpoint', 'control_setpoint', 'Q_irr', 'T_out']


def plot_bo_results(opt, total_cost_list, best_obj_list):
    for i in range(opt.opt_problem.num_constrs):
        plt.figure()
        plt.plot(np.array(total_cost_list)[:, i])
        plt.xlabel('Optimization step')
        plt.ylabel('Total violation cost')
        plt.title(opt.opt_problem.config['problem_name'])

    plt.figure()
    plt.plot(best_obj_list)
    plt.xlabel('Optimization step')
    plt.ylabel('Best feasible objective found')
    plt.title(opt.opt_problem.config['problem_name'])


# get_optimizer: construct the problem
def get_optimizer(problem_name, optimizer_type, optimizer_config,
                  init_points_id=0, discomfort_thr=2.0, vars_to_fix=
                  ['high_on_time', 'high_off_time', 'high_setpoint',
               'low_setpoint', 'control_setpoint', 'Q_irr', 'T_out'],
                  start_date_time=None, fixed_param=None,
                  discomfort_weight=0.01, tune_var_scale='log'):
    problem_config = util.get_config(
        problem_name, gp_kernel='Matern52', init_points_id=init_points_id,
        discomfort_thr=discomfort_thr, vars_to_fix=vars_to_fix,
        start_eval_time=start_date_time, room_simulator='PCNN',
        discomfort_weight=discomfort_weight, tune_PI_scale=tune_var_scale)
    if fixed_param is not None:
        problem_config['init_safe_points'] = fixed_param
    problem = vabo.optimization_problem.OptimizationProblem(problem_config)
    if optimizer_type == 'safe_bo':
        opt = vabo.safe_optimizer.SafeBO(problem, optimizer_config)
        best_obj_list = [-opt.best_obj]
    if optimizer_type == 'constrained_bo':
        opt = vabo.constrained_bo.ConstrainedBO(problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    if optimizer_type == 'violation_aware_bo':
        opt = vabo.violation_aware_bo.ViolationAwareBO(
            problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    if optimizer_type == 'no opt':
        opt = KeepDefaultOpt(problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    if optimizer_type == 'grid search':
        opt = GridSearchOpt(problem, optimizer_config)
        best_obj_list = [opt.best_obj]

    total_cost_list = [opt.cumu_vio_cost]
    return opt, best_obj_list, total_cost_list


def evaluate_one_fixed_control(fixed_param, discomfort_thr,
                               discomfort_weight=0.01):
    # try fixing one parameter
    no_opt_config = copy.deepcopy(optimizer_base_config)

    no_opt, no_opt_best_obj_list, no_opt_total_cost_list = get_optimizer(
        no_opt_config['problem_name'], 'no opt', no_opt_config,
        discomfort_thr=discomfort_thr, vars_to_fix=vars_to_fix,
        fixed_param=fixed_param, discomfort_weight=discomfort_weight)
    for _ in range(optimization_config['eval_budget']):
        y_obj, constr_vals = no_opt.make_step()
    simulator = no_opt.opt_problem.simulator
    simulator.update_history_dict()
    cumulative_discomfort = simulator.cumulative_discomfort
    date_time_list = list(simulator.history_dict.keys())
    cumulative_energy = sum(
        simulator.history_dict_to_list('power room', date_time_list)
    ) * 0.001 * 0.25
    num_of_data = len(date_time_list)
    energy_per_day = cumulative_energy / (num_of_data / 96)
    ave_discomfort = cumulative_discomfort / num_of_data
    return energy_per_day, ave_discomfort


class OptimizerEvaluator:

    def __init__(self):
        self.opt_result_dict = None
        self.obj_list_dict = None
        self.constrain_list_dict = None
        self.energy_list_dict = None
        self.discomfort_list_dict = None
        self.seasonal_energy_list_dict = None
        self.seasonal_discomfort_list_dict = None
        self.evaluated_points_list_dict = None

    def evaluate_one_optimizer(self, opt_config, optimizer_type,
                               discomfort_weight=0.01):
        opt_result_dict = dict()

        obj_list_dict = dict()
        constrain_list_dict = dict()
        energy_list_dict = dict()
        discomfort_list_dict = dict()

        seasonal_energy_list_dict = dict()
        seasonal_discomfort_list_dict = dict()
        evaluated_points_list_dict = dict()
        for discomfort_thr in discomfort_thr_list:
            opt, opt_best_obj_list, opt_total_cost_list = get_optimizer(
                opt_config['problem_name'], optimizer_type, opt_config,
                discomfort_thr=discomfort_thr, vars_to_fix=vars_to_fix,
                discomfort_weight=discomfort_weight)
            opt_obj_list = []
            constraints_list = []
            energy_list = []
            discomfort_list = []
            seasonal_energy_list = []
            seasonal_discomfort_list = []
            for _ in range(optimization_config['eval_budget']):
                y_obj, constr_vals = opt.make_step()
                opt_total_cost_list.append(opt.cumu_vio_cost)
                opt_best_obj_list.append(opt.best_obj)
                opt_obj_list.append(y_obj)
                constraints_list.append(constr_vals)
                energy, discomfort = opt.opt_problem.simulator.\
                    get_recent_energy_discomfort_per_day()
                energy_list.append(energy)
                discomfort_list.append(discomfort)
                fixed_param = opt.opt_problem.evaluated_points_list[-1]
                seasonal_energy, seasonal_discomfort = \
                    evaluate_one_fixed_control(
                        fixed_param, discomfort_thr,
                        discomfort_weight=discomfort_weight)
                seasonal_energy_list.append(seasonal_energy)
                seasonal_discomfort_list.append(seasonal_discomfort)

            opt_result_dict[discomfort_thr] = opt
            obj_list_dict[discomfort_thr] = opt_obj_list
            constrain_list_dict[discomfort_thr] = constraints_list
            energy_list_dict[discomfort_thr] = energy_list
            discomfort_list_dict[discomfort_thr] = discomfort_list

            seasonal_energy_list_dict[discomfort_thr] = seasonal_energy_list
            seasonal_discomfort_list_dict[discomfort_thr] = \
                seasonal_discomfort_list
            evaluated_points_list_dict[discomfort_thr] = opt.opt_problem.\
                evaluated_points_list
        self.opt_result_dict = opt_result_dict
        self.obj_list_dict = obj_list_dict
        self.constrain_list_dict = constrain_list_dict
        self.energy_list_dict = energy_list_dict
        self.discomfort_list_dict = discomfort_list_dict
        self.seasonal_energy_list_dict = seasonal_energy_list_dict
        self.seasonal_discomfort_list_dict = seasonal_discomfort_list_dict
        self.evaluated_points_list_dict = evaluated_points_list_dict

    def save_result(self, save_path):
        np.savez(save_path, self.obj_list_dict, self.constrain_list_dict,
                 self.energy_list_dict, self.discomfort_list_dict,
                 self.seasonal_energy_list_dict,
                 self.seasonal_discomfort_list_dict,
                 self.evaluated_points_list_dict)

tune_var_scale = 'log_'
discomfort_weight = 0.1
save_name_append = f'_{discomfort_weight}_{tune_var_scale}'

safe_bo_config = copy.deepcopy(optimizer_base_config)

#safe_bo_evalutor = OptimizerEvaluator()
#safe_bo_evalutor.evaluate_one_optimizer(safe_bo_config, 'safe_bo',
#                                        discomfort_weight)
#safe_bo_evalutor.save_result(f'./result/safe_bo'+save_name_append)

grid_search_config = copy.deepcopy(safe_bo_config)
grid_search_config.update({
    'kernel_type': 'Gaussian',
})
grid_search_evaluator = OptimizerEvaluator()
grid_search_evaluator.evaluate_one_optimizer(grid_search_config,
                                                'grid search',
                                                discomfort_weight)
grid_search_evaluator.save_result(
    f'./result/grid_search'+save_name_append)


constrained_bo_config = copy.deepcopy(safe_bo_config)
constrained_bo_config.update({
    'kernel_type': 'Gaussian',
})
constrained_bo_evaluator = OptimizerEvaluator()
constrained_bo_evaluator.evaluate_one_optimizer(constrained_bo_config,
                                                'constrained_bo',
                                                discomfort_weight)
constrained_bo_evaluator.save_result(
    f'./result/constrained_bo'+save_name_append)


violation_aware_bo_config = copy.deepcopy(optimizer_base_config)
violation_aware_bo_config.update({
        'single_max_budget': 3,
        'total_vio_budgets': np.array([1.0, 20.0]),
        'prob_eps': 5e-2,
        'beta_0': 1,
        'total_eval_num': optimization_config['eval_budget'],
})
violation_aware_bo_evaluator = OptimizerEvaluator()
violation_aware_bo_evaluator.evaluate_one_optimizer(
    violation_aware_bo_config,
    'violation_aware_bo',
    discomfort_weight=discomfort_weight
)
violation_aware_bo_evaluator.save_result(
    f'./result/violation_aware_bo'+save_name_append)
