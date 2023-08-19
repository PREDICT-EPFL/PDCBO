import copy
import os
import pickle
import datetime
from tune_util import get_vacbo_optimizer


# parameter configurations to enumerate
energy_thr_list = [18, 19, 20, 21]  # [15, 16, 17, 18]  # [19, 20, 21, 22]

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
optimization_config = {
    'eval_budget': 300
}
optimizer_base_config = {
    'noise_level': [0.75/10, 0.002/10, 1.39/10],
    'kernel_var': 1.0,
    'problem_name': 'SinglePIRoomEvaluator',
    'normalize_input': False
}
VARS_TO_FIX = ['high_off_time', 'low_setpoint',
               'control_setpoint']
CONTEXTUAL_VARS = ['Q_irr', 'T_out', 'T_init']
tune_var_scale = 'log'
save_name_append = f'_{tune_var_scale}_with_context'
TUNE_OBJ = 'discomfort'
discomfort_weight = 0.0

energy_thr_to_ave_energy_pdcbo = dict()
energy_thr_to_ave_discomfort_pdcbo = dict()
energy_thr_to_energy_pdcbo = dict()
energy_thr_to_discomfort_pdcbo = dict()
energy_thr_to_history_dict_pdcbo = dict()

energy_thr_to_ave_energy_cbo = dict()
energy_thr_to_ave_discomfort_cbo = dict()
energy_thr_to_energy_cbo = dict()
energy_thr_to_discomfort_cbo = dict()
energy_thr_to_history_dict_cbo = dict()

energy_thr_to_ave_energy_safe_bo = dict()
energy_thr_to_ave_discomfort_safe_bo = dict()
energy_thr_to_energy_safe_bo = dict()
energy_thr_to_discomfort_safe_bo = dict()
energy_thr_to_history_dict_safe_bo = dict()


def run_opt(bo_config, optimizer_type):
    energy_thr_to_ave_energy = dict()
    energy_thr_to_ave_discomfort = dict()
    energy_thr_to_energy = dict()
    energy_thr_to_discomfort = dict()
    energy_thr_to_history_dict = dict()

    for energy_thr in energy_thr_list:
        divided_energy_thr = energy_thr / 1.0
        opt, opt_total_cost_list, opt_problem = get_vacbo_optimizer(
                    bo_config['problem_name'], optimizer_type, bo_config,
                    vars_to_fix=VARS_TO_FIX,
                    contextual_vars=CONTEXTUAL_VARS,
                    discomfort_weight=discomfort_weight, tune_obj=TUNE_OBJ,
                    energy_thr=divided_energy_thr
        )
        opt_obj_list = []
        constraints_list = []
        energy_list = []
        discomfort_list = []
        average_energy_list = []
        average_discomfort_list = []

        for _ in range(optimization_config['eval_budget']):
            context_vars = opt_problem.get_context(opt_problem.simulator)
            y_obj, constr_vals = opt.make_step(context_vars)
            if optimizer_type == 'safe_bo':
                new_cumu_cost = opt.safe_bo.cumu_vio_cost
                opt_problem = opt.safe_bo.opt_problem
            if optimizer_type == 'constrained_bo':
                new_cumu_cost = opt.constrained_bo.cumu_vio_cost
                opt_problem = opt.constrained_bo.opt_problem
            if optimizer_type == 'violation_aware_bo':
                new_cumu_cost = opt.violation_aware_bo.cumu_vio_cost
                opt_problem = opt.violation_aware_bo.opt_problem
            if optimizer_type == 'pdcbo':
                new_cumu_cost = opt.pdbo.cumu_vio_cost
                opt_problem = opt.pdbo.opt_problem
            if optimizer_type == 'no opt':
                new_cumu_cost = opt.cumu_vio_cost
                opt_problem = opt.opt_problem
            if optimizer_type == 'grid search':
                new_cumu_cost = opt.cumu_vio_cost
                opt_problem = opt.opt_problem

            opt_total_cost_list.append(new_cumu_cost)
            opt_obj_list.append(y_obj)
            constraints_list.append(constr_vals)
            energy, discomfort = \
                opt_problem.simulator.get_recent_energy_discomfort_per_day()

            energy_list.append(energy)
            discomfort_list.append(discomfort)
            print_log = True
            if print_log:
                print(f"For {opt_problem.problem_name}, in step {_}, " +
                      "with energy threshold " +
                      f"{divided_energy_thr}, we get energy {energy}" +
                      f" and discomfort {discomfort}, with the point "
                      + f" {opt_problem.evaluated_points_list[-1]}.")
            average_energy_list.append(
                opt_problem.simulator.cumulative_energy / 96.0 /
                len(opt_problem.evaluated_constrs_list)
            )
            average_discomfort_list.append(
                opt_problem.simulator.cumulative_discomfort * 0.25 /
                len(opt_problem.evaluated_constrs_list)
            )

        energy_thr_to_ave_discomfort[energy_thr] = \
            average_discomfort_list
        energy_thr_to_ave_energy[energy_thr] = average_energy_list

        energy_thr_to_discomfort[energy_thr] = discomfort_list
        energy_thr_to_energy[energy_thr] = energy_list
        energy_thr_to_history_dict[energy_thr] = opt_problem.simulator.\
            history_dict
    return energy_thr_to_discomfort, energy_thr_to_energy, \
        energy_thr_to_ave_discomfort, energy_thr_to_ave_energy, \
        energy_thr_to_history_dict


# run PDCBO
pdcbo_config = copy.deepcopy(optimizer_base_config)
pdcbo_config.update({
        'eta_0': 1.0,
        'eta_func': lambda t: 1.0,
        'total_eval_num': optimization_config['eval_budget'],
        'init_dual': 1.0,
        'lcb_coef': lambda t: 1.0  # 1e-6
    })

optimizer_type = 'pdcbo'
energy_thr_to_discomfort_pdcbo, energy_thr_to_energy_pdcbo, \
    energy_thr_to_ave_discomfort_pdcbo, \
    energy_thr_to_ave_energy_pdcbo, energy_thr_to_history_dict_pdcbo \
    = run_opt(pdcbo_config, optimizer_type)


cbo_config = copy.deepcopy(optimizer_base_config)
cbo_config.update({
        'total_eval_num': optimization_config['eval_budget'],
        'kernel_type': 'Gaussian'
    })

optimizer_type = 'constrained_bo'
energy_thr_to_discomfort_cbo, energy_thr_to_energy_cbo, \
    energy_thr_to_ave_discomfort_cbo, \
    energy_thr_to_ave_energy_cbo, energy_thr_to_history_dict_cbo \
    = run_opt(cbo_config, optimizer_type)


safebo_config = copy.deepcopy(optimizer_base_config)
safebo_config.update({
        'total_eval_num': optimization_config['eval_budget'],
        'kernel_type': 'Gaussian'
    })
optimizer_type = 'safe_bo'
energy_thr_to_discomfort_safe_bo, energy_thr_to_energy_safe_bo, \
    energy_thr_to_ave_discomfort_safe_bo, \
    energy_thr_to_ave_energy_safe_bo, energy_thr_to_history_dict_safe_bo \
    = run_opt(safebo_config, optimizer_type)

now_time_str = datetime.datetime.now().strftime(
        "%H_%M_%S-%b_%d_%Y")

with open(f'./result/energy_constrained_discomfort_min_{now_time_str}.pkl',
          'wb') as f:
    pickle.dump([
        energy_thr_to_ave_energy_pdcbo,
        energy_thr_to_ave_discomfort_pdcbo,
        energy_thr_to_energy_pdcbo,
        energy_thr_to_discomfort_pdcbo,
        energy_thr_to_ave_energy_cbo,
        energy_thr_to_ave_discomfort_cbo,
        energy_thr_to_energy_cbo,
        energy_thr_to_discomfort_cbo,
        energy_thr_to_ave_energy_safe_bo,
        energy_thr_to_ave_discomfort_safe_bo,
        energy_thr_to_energy_safe_bo,
        energy_thr_to_discomfort_safe_bo,
        energy_thr_to_history_dict_pdcbo,
        energy_thr_to_history_dict_cbo,
        energy_thr_to_history_dict_safe_bo
    ], f)
