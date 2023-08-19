import copy
import os
import pickle
import datetime
import util
from tune_util import get_vacbo_optimizer


# parameter configurations to enumerate
discomfort_thr_list = [5, 10, 15, 20]

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
optimization_config = {
    'eval_budget': 300
}


steps_per_discomfort_thr = int(optimization_config['eval_budget'] /
        len(discomfort_thr_list)
        )


optimizer_base_config = {
    'noise_level': [0.002/5, 0.75/5, 1.39/5],
    'kernel_var': 1.0,
    'problem_name': 'SinglePIRoomEvaluator',
    'normalize_input': False
}
VARS_TO_FIX = ['high_off_time', 'low_setpoint',
               'control_setpoint']
CONTEXTUAL_VARS = ['Q_irr', 'T_out', 'T_init']
tune_var_scale = 'log'
save_name_append = f'_{tune_var_scale}_with_context'
TUNE_OBJ = 'energy'
ave_discomfort_range = (0, 20)
ave_energy_range = (16, 23)
discomfort_weight = 0.0

discomfort_thr_to_ave_energy_pdcbo = dict()
discomfort_thr_to_ave_discomfort_pdcbo = dict()
discomfort_thr_to_energy_pdcbo = dict()
discomfort_thr_to_discomfort_pdcbo = dict()
discomfort_thr_to_discomfort_thr_pdcbo = dict()
discomfort_thr_to_history_dict_pdcbo = dict()


discomfort_thr_to_ave_energy_cbo = dict()
discomfort_thr_to_ave_discomfort_cbo = dict()
discomfort_thr_to_energy_cbo = dict()
discomfort_thr_to_discomfort_cbo = dict()
discomfort_thr_to_discomfort_thr_cbo = dict()
discomfort_thr_to_history_dict_cbo = dict()

discomfort_thr_to_ave_energy_safe_bo = dict()
discomfort_thr_to_ave_discomfort_safe_bo = dict()
discomfort_thr_to_energy_safe_bo = dict()
discomfort_thr_to_discomfort_safe_bo = dict()
discomfort_thr_to_discomfort_thr_safe_bo = dict()
discomfort_thr_to_history_dict_safe_bo = dict()


def run_opt(bo_config, optimizer_type):
    discomfort_thr_to_ave_energy = dict()
    discomfort_thr_to_ave_discomfort = dict()
    discomfort_thr_to_energy = dict()
    discomfort_thr_to_discomfort = dict()
    discomfort_thr_to_discomfort_thr = dict()
    discomfort_thr_to_history_dict = dict()
    for discomfort_thr in [discomfort_thr_list[0]]:
        opt, opt_total_cost_list, opt_problem = get_vacbo_optimizer(
                    bo_config['problem_name'], optimizer_type, bo_config,
                    discomfort_thr=discomfort_thr, vars_to_fix=VARS_TO_FIX,
                    contextual_vars=CONTEXTUAL_VARS,
                    discomfort_weight=discomfort_weight, tune_obj=TUNE_OBJ)
        opt_obj_list = []
        constraints_list = []
        energy_list = []
        discomfort_list = []
        step_discomfort_thr_list = [] 
        average_energy_list = []
        average_discomfort_list = []
        
        prev_discomfort_thr = discomfort_thr
        for _ in range(optimization_config['eval_budget']):
            step_discomfort_thr_id = max(
                    min(int(_/steps_per_discomfort_thr), len(discomfort_thr_list)-1),
                    0
                    )
            step_discomfort_thr = discomfort_thr_list[step_discomfort_thr_id]
            step_config = util.get_config(
                    optimizer_base_config['problem_name'],
                    gp_kernel='Gaussian', 
                    discomfort_thr=step_discomfort_thr, 
                    vars_to_fix=VARS_TO_FIX,
                    start_eval_time=None, 
                    room_simulator='PCNN',
                    discomfort_weight=discomfort_weight, 
                    tune_PI_scale='log',
                    contextual_vars=CONTEXTUAL_VARS, 
                    tune_obj=TUNE_OBJ,
                    energy_thr=0.013
                    )
                      
            context_vars = opt_problem.get_context(opt_problem.simulator)
            if optimizer_type == 'safe_bo':
                base_opt = opt.safe_bo
                opt_problem = opt.safe_bo.opt_problem
            if optimizer_type == 'constrained_bo':
                base_opt = opt.constrained_bo
                opt_problem = opt.constrained_bo.opt_problem
            if optimizer_type == 'violation_aware_bo': 
                base_opt = opt.violation_aware_bo
                opt_problem = opt.violation_aware_bo.opt_problem
            if optimizer_type == 'pdcbo':
                base_opt = opt.pdbo
                opt_problem = opt.pdbo.opt_problem
            if optimizer_type == 'no opt':
                base_opt = opt
                opt_problem = opt.opt_problem
            if optimizer_type == 'grid search':
                base_opt = opt 
                opt_problem = opt.opt_problem

            opt_problem.obj = step_config['obj']
            opt_problem.constrs_list = step_config['constrs_list']
            
            normalized_prev_discomfort_thr = prev_discomfort_thr / 24.0
            normalized_step_discomfort_thr = step_discomfort_thr / 24.0
            if optimizer_type == 'safe_bo':
                discomfort_gp = base_opt.opt.gps[1] 
                base_opt.opt.gps[1].set_XY(
                        discomfort_gp.X,
                        discomfort_gp.Y - normalized_prev_discomfort_thr +
                        normalized_step_discomfort_thr
                        )
            elif optimizer_type in ['constrained_bo', 'violation_aware_bo',
                    'pdcbo']:
                discomfort_gp = base_opt.gp_constr_list[0]
                base_opt.gp_constr_list[0].set_XY(
                        discomfort_gp.X,
                        discomfort_gp.Y + normalized_prev_discomfort_thr -
                        normalized_step_discomfort_thr
                        )
            
            prev_discomfort_thr = step_discomfort_thr
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
            energy, discomfort = \
                opt_problem.simulator.get_recent_energy_discomfort_per_day()

            energy_list.append(energy)
            discomfort_list.append(discomfort)
            step_discomfort_thr_list.append(step_discomfort_thr)
            print_log = True
            if print_log:
                print(f"For {opt_problem.problem_name}, in step {_}, " +
                      "with discomfort threshold " +
                      f"{discomfort_thr}, we get energy {energy}" +
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

        discomfort_thr_to_ave_discomfort[discomfort_thr] = \
            average_discomfort_list
        discomfort_thr_to_ave_energy[discomfort_thr] = average_energy_list

        discomfort_thr_to_discomfort[discomfort_thr] = discomfort_list
        discomfort_thr_to_discomfort_thr[discomfort_thr] = step_discomfort_thr_list
        discomfort_thr_to_energy[discomfort_thr] = energy_list
        discomfort_thr_to_history_dict[discomfort_thr] = \
            opt_problem.simulator.history_dict

    return discomfort_thr_to_discomfort, discomfort_thr_to_energy, \
        discomfort_thr_to_ave_discomfort, discomfort_thr_to_ave_energy, \
        discomfort_thr_to_history_dict, discomfort_thr_to_discomfort_thr


# run PDCBO
pdcbo_config = copy.deepcopy(optimizer_base_config)
pdcbo_config.update({
        'eta_0': 1.0 / 300.0,
        'eta_func': lambda t: 1.0 / 300.0,
        'total_eval_num': optimization_config['eval_budget'],
        'init_dual': 10.0 / 300.0,
        'lcb_coef': lambda t: 1.0  # 1e-6
    })

optimizer_type = 'pdcbo'
discomfort_thr_to_discomfort_pdcbo, discomfort_thr_to_energy_pdcbo, \
    discomfort_thr_to_ave_discomfort_pdcbo, \
    discomfort_thr_to_ave_energy_pdcbo, discomfort_thr_to_history_dict_pdcbo, \
    discomfort_thr_to_discomfort_thr_pdcbo \
    = run_opt(pdcbo_config, optimizer_type)


cbo_config = copy.deepcopy(optimizer_base_config)
cbo_config.update({
        'total_eval_num': optimization_config['eval_budget'],
        'kernel_type': 'Gaussian'
    })

optimizer_type = 'constrained_bo'
discomfort_thr_to_discomfort_cbo, discomfort_thr_to_energy_cbo, \
    discomfort_thr_to_ave_discomfort_cbo, \
    discomfort_thr_to_ave_energy_cbo, discomfort_thr_to_history_dict_cbo, \
    discomfort_thr_to_discomfort_thr_cbo \
    = run_opt(cbo_config, optimizer_type)


safebo_config = copy.deepcopy(optimizer_base_config)
safebo_config.update({
        'total_eval_num': optimization_config['eval_budget'],
        'kernel_type': 'Gaussian'
    })
optimizer_type = 'safe_bo'
discomfort_thr_to_discomfort_safe_bo, discomfort_thr_to_energy_safe_bo, \
    discomfort_thr_to_ave_discomfort_safe_bo, \
    discomfort_thr_to_ave_energy_safe_bo, \
    discomfort_thr_to_history_dict_safe_bo, \
    discomfort_thr_to_discomfort_thr_safe_bo = run_opt(
        safebo_config, optimizer_type)
now_time_str = datetime.datetime.now().strftime(
        "%H_%M_%S-%b_%d_%Y")


with open(f'./result/discomfort_constrained_energy_min_{now_time_str}.pkl',
          'wb') as f:
    pickle.dump([
        discomfort_thr_to_ave_energy_pdcbo,
        discomfort_thr_to_ave_discomfort_pdcbo,
        discomfort_thr_to_energy_pdcbo,
        discomfort_thr_to_discomfort_pdcbo,
        discomfort_thr_to_ave_energy_cbo,
        discomfort_thr_to_ave_discomfort_cbo,
        discomfort_thr_to_energy_cbo,
        discomfort_thr_to_discomfort_cbo,
        discomfort_thr_to_ave_energy_safe_bo,
        discomfort_thr_to_ave_discomfort_safe_bo,
        discomfort_thr_to_energy_safe_bo,
        discomfort_thr_to_discomfort_safe_bo,
        discomfort_thr_to_history_dict_pdcbo,
        discomfort_thr_to_history_dict_cbo,
        discomfort_thr_to_history_dict_safe_bo,
        discomfort_thr_to_discomfort_thr_pdcbo,
        discomfort_thr_to_discomfort_thr_cbo,
        discomfort_thr_to_discomfort_thr_safe_bo
    ], f)
