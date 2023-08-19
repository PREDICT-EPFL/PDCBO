import safe_optimizer
import optimization_problem
import constrained_bo
import pdbo
from lcb2 import LCB2
from lcb2 import EPBO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_optimizer(optimizer_type, optimizer_config, problem_config):
    problem = optimization_problem.OptimizationProblem(problem_config)
    if optimizer_type == 'safe_bo':
        opt = safe_optimizer.SafeBO(problem, optimizer_config)
        best_obj_list = [-opt.best_obj]
    if optimizer_type == 'constrained_bo':
        opt = constrained_bo.ConstrainedBO(problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    if optimizer_type == 'pdbo':
        opt = pdbo.pd_bo.PDBO(problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    if optimizer_type == 'lcb2':
        opt = LCB2(problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    if optimizer_type == 'epbo':
        opt = EPBO(problem, optimizer_config)
        best_obj_list = [opt.best_obj]

    total_cost_list = [opt.cumu_vio_cost]
    return opt, best_obj_list, total_cost_list


def get_init_obj_constrs(opt):
    init_obj_val_arr, init_constr_val_arr = \
            opt.get_obj_constr_val(opt.x0_arr)
    init_obj_val_list = [init_obj_val_arr[0, 0]]
    init_constr_val_list = [init_constr_val_arr[0, :].tolist()]
    return init_obj_val_list, init_constr_val_list


def get_safe_bo_result(problem_config, safe_bo_config):
    safe_opt, safe_bo_best_obj_list, safe_bo_total_cost_list = \
        get_optimizer('safe_bo', safe_bo_config, problem_config)

    init_obj_val_list, init_constr_val_list = get_init_obj_constrs(safe_opt)

    safe_opt_obj_list = init_obj_val_list
    safe_opt_constr_list = init_constr_val_list
    for _ in range(safe_bo_config['eval_budget']):
        y_obj, constr_vals = safe_opt.make_step()
        safe_bo_total_cost_list.append(safe_opt.cumu_vio_cost)
        safe_bo_best_obj_list.append(-safe_opt.best_obj)
        safe_opt_obj_list.append(y_obj[0, 0])
        safe_opt_constr_list.append([constr_vals[0, 0], constr_vals[0, 1]])
    return safe_bo_total_cost_list, safe_bo_best_obj_list, safe_opt, \
        safe_opt_obj_list, safe_opt_constr_list


# test ConstrainedBO on the test function
def get_constrained_bo_result(problem_config, constrained_bo_config):
    constrained_opt, constrained_bo_best_obj_list, \
        constrained_bo_total_cost_list = get_optimizer(
            'constrained_bo', constrained_bo_config, problem_config)

    init_obj_val_list, init_constr_val_list = get_init_obj_constrs(
        constrained_opt)
    constrained_opt_obj_list = init_obj_val_list
    constrained_opt_constr_list = init_constr_val_list
    for _ in range(constrained_bo_config['eval_budget']):
        y_obj, constr_vals = constrained_opt.make_step()
        constrained_bo_total_cost_list.append(constrained_opt.cumu_vio_cost)
        constrained_bo_best_obj_list.append(constrained_opt.best_obj)
        constrained_opt_obj_list.append(y_obj[0, 0])
        constrained_opt_constr_list.append(
            [constr_vals[0, 0], constr_vals[0, 1]])
    return constrained_bo_total_cost_list, constrained_bo_best_obj_list, \
        constrained_opt, constrained_opt_obj_list, constrained_opt_constr_list


# test LCB2 on the test function
def get_lcb2_result(problem_config, base_opt_config):
    lcb2_config = base_opt_config.copy()
    lcb2_config.update({
        'total_eval_num': base_opt_config['eval_budget'],
        }
    )
    lcb2_opt, lcb2_best_obj_list, lcb2_total_cost_list = get_optimizer(
            'lcb2', lcb2_config, problem_config)
    lcb2_opt_obj_list = lcb2_opt.init_obj_val_list
    lcb2_opt_constr_list = lcb2_opt.init_constr_val_list
    for _ in range(lcb2_config['eval_budget']):
        print(f'lcb2 step {_}.')
        y_obj, constr_vals = lcb2_opt.make_step()
        lcb2_total_cost_list.append(
            lcb2_opt.cumu_vio_cost)
        lcb2_best_obj_list.append(lcb2_opt.best_obj)
        lcb2_opt_obj_list.append(y_obj[0, 0])
        lcb2_opt_constr_list.append([constr_vals[0, 0], constr_vals[0, 1]])
    return lcb2_total_cost_list, \
        lcb2_best_obj_list, lcb2_opt, \
        lcb2_opt_obj_list, lcb2_opt_constr_list

# test LCB2 on the test function
def get_epbo_result(problem_config, base_opt_config):
    epbo_config = base_opt_config.copy()
    epbo_config.update({
        'total_eval_num': base_opt_config['eval_budget'],
        }
    )
    epbo_opt, epbo_best_obj_list, epbo_total_cost_list = get_optimizer(
            'epbo', epbo_config, problem_config)
    epbo_opt_obj_list = epbo_opt.init_obj_val_list
    epbo_opt_constr_list = epbo_opt.init_constr_val_list
    for _ in range(epbo_config['eval_budget']):
        print(f'epbo step {_}.')
        y_obj, constr_vals = epbo_opt.make_step()
        epbo_total_cost_list.append(
            epbo_opt.cumu_vio_cost)
        epbo_best_obj_list.append(epbo_opt.best_obj)
        epbo_opt_obj_list.append(y_obj[0, 0])
        epbo_opt_constr_list.append([constr_vals[0, 0], constr_vals[0, 1]])
    return epbo_total_cost_list, \
        epbo_best_obj_list, epbo_opt, \
        epbo_opt_obj_list, epbo_opt_constr_list

def get_pdbo_result(problem_config, base_opt_config):
    pdbo_config = base_opt_config.copy()
    pdbo_config.update({
        'beta_0': 3,
        'eta_0': 2,
        'total_eval_num': base_opt_config['eval_budget'],
        'normalize_input': False
        })
    pdbo_opt, pdbo_best_obj_list, \
        pdbo_total_cost_list = get_optimizer(
            'pdbo', pdbo_config, problem_config)
    init_obj_val_list, init_constr_val_list = get_init_obj_constrs(
        pdbo_opt)

    pdbo_obj_list = init_obj_val_list
    pdbo_constr_val_list = init_constr_val_list
    for _ in range(pdbo_config['eval_budget']):
        y_obj, constr_vals = pdbo_opt.make_step()
        pdbo_total_cost_list.append(
            pdbo_opt.cumu_vio_cost)
        pdbo_best_obj_list.append(pdbo_opt.best_obj)
        pdbo_obj_list.append(y_obj[0, 0])
        pdbo_constr_val_list.append([constr_vals[0, 0], constr_vals[0, 1]])
    return pdbo_total_cost_list, pdbo_best_obj_list, pdbo_opt, \
        pdbo_obj_list, pdbo_constr_val_list
