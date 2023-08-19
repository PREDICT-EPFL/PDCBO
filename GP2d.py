import copy
import os
import pickle
import datetime
import util
from tune_util import get_vacbo_optimizer


# parameter configurations to enumerate

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
optimization_config = {
    'eval_budget': 1000
}
run_traj_num = 50
optimizer_base_config = {
    'noise_level': [0.001, 0.001],
    'kernel_var': 1.0,
    'problem_name': 'sample_GP_two_dim',
    'normalize_input': False
}
VARS_TO_FIX = []
CONTEXTUAL_VARS = ['z']

save_name_append = f'_with_context'


def run_opt(bo_config, optimizer_type, problem_config):
    print('Start running the opt problem.')
    opt, opt_total_cost_list, opt_problem = get_vacbo_optimizer(
        bo_config['problem_name'], optimizer_type, bo_config,
        vars_to_fix=VARS_TO_FIX,
        contextual_vars=CONTEXTUAL_VARS,
        problem_config=problem_config
    )

    print('The opt problem got.')
    opt_obj_list = []
    constraints1_list = []
    contexts_list = []
    cond_min_list = []
    for _ in range(optimization_config['eval_budget']):
        context_vars, cond_min = opt_problem.get_context(step=_)
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

        opt_obj_list.append(y_obj[0,0])
        constraints1_list.append(constr_vals[:, 0])
        contexts_list.append(context_vars)
        cond_min_list.append(cond_min)
        print_log = True
        if print_log:
            print(f"For the problem {opt_problem.problem_name}, in \
                  step {_}, we get objective {y_obj} and constraints \
                  {constr_vals}.")

    return opt_obj_list, constraints1_list, \
        opt_problem.evaluated_points_list, contexts_list, cond_min_list

opt_obj_list_pdcbo_list = []
constraints1_list_pdcbo_list = []
evaluated_points_list_pdcbo_list = []
contexts_list_pdcbo_list = []
cond_min_list_pdcbo_list = []
opt_obj_list_cbo_list = []
constraints1_list_cbo_list = []
evaluated_points_list_cbo_list = []
contexts_list_cbo_list = []
cond_min_list_cbo_list = []
opt_obj_list_safebo_list = []
constraints1_list_safebo_list = []
evaluated_points_list_safebo_list = []
contexts_list_safebo_list = []
cond_min_list_safebo_list = []

for traj_id in range(run_traj_num):
    problem_name = 'sample_GP_two_dim'
    problem_config = util.get_config(
                problem_name, gp_kernel='Gaussian',
                vars_to_fix=VARS_TO_FIX, contextual_vars=CONTEXTUAL_VARS)

    # run PDCBO
    pdcbo_config = copy.deepcopy(optimizer_base_config)
    pdcbo_config.update({
        'eta_0': 1.0,
        'eta_func': lambda t: 1.0,
        'total_eval_num': optimization_config['eval_budget'],
        'init_dual': 0.0,
        'lcb_coef': lambda t: 1.0  # 1e-6
    })

    optimizer_type = 'pdcbo'
    opt_obj_list_pdcbo, constraints1_list_pdcbo,  \
        evaluated_points_list_pdcbo, contexts_list_pdcbo, cond_min_list_pdcbo \
        = run_opt(pdcbo_config, optimizer_type, problem_config)

    cbo_config = copy.deepcopy(optimizer_base_config)
    cbo_config.update({
            'total_eval_num': optimization_config['eval_budget'],
            'kernel_type': 'Gaussian'
    })

    optimizer_type = 'constrained_bo'
    opt_obj_list_cbo, constraints1_list_cbo, \
        evaluated_points_list_cbo, contexts_list_cbo, cond_min_list_cbo \
        = run_opt(cbo_config, optimizer_type, problem_config)


    safebo_config = copy.deepcopy(optimizer_base_config)
    safebo_config.update({
        'total_eval_num': optimization_config['eval_budget'],
        'kernel_type': 'Gaussian'
    })

    optimizer_type = 'safe_bo'
    opt_obj_list_safebo, constraints1_list_safebo,\
        evaluated_points_list_safebo, contexts_list_safebo, \
        cond_min_list_safebo \
        = run_opt(safebo_config, optimizer_type, problem_config)

    opt_obj_list_pdcbo_list.append(opt_obj_list_pdcbo)
    constraints1_list_pdcbo_list.append(constraints1_list_pdcbo)
    evaluated_points_list_pdcbo_list.append(evaluated_points_list_pdcbo)
    contexts_list_pdcbo_list.append(contexts_list_pdcbo)
    cond_min_list_pdcbo_list.append(cond_min_list_pdcbo)

    opt_obj_list_cbo_list.append(opt_obj_list_cbo)
    constraints1_list_cbo_list.append(constraints1_list_cbo)
    evaluated_points_list_cbo_list.append(evaluated_points_list_cbo)
    contexts_list_cbo_list.append(contexts_list_cbo)
    cond_min_list_cbo_list.append(cond_min_list_cbo)

    opt_obj_list_safebo_list.append(opt_obj_list_safebo)
    constraints1_list_safebo_list.append(constraints1_list_safebo)
    evaluated_points_list_safebo_list.append(evaluated_points_list_safebo)
    contexts_list_safebo_list.append(contexts_list_safebo)
    cond_min_list_safebo_list.append(cond_min_list_safebo)

now_time_str = datetime.datetime.now().strftime(
        "%H_%M_%S-%b_%d_%Y")

with open(f'./result/twod_gp_min_{now_time_str}.pkl',
          'wb') as f:
    pickle.dump([
        opt_obj_list_pdcbo_list,
        constraints1_list_pdcbo_list,
        evaluated_points_list_pdcbo_list,
        contexts_list_pdcbo_list,
        cond_min_list_pdcbo_list,
        opt_obj_list_cbo_list,
        constraints1_list_cbo_list,
        evaluated_points_list_cbo_list,
        contexts_list_cbo_list,
        cond_min_list_cbo_list,
        opt_obj_list_safebo_list,
        constraints1_list_safebo_list,
        evaluated_points_list_safebo_list,
        contexts_list_safebo_list,
        cond_min_list_safebo_list
    ], f)
