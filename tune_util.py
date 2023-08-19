import util
import vabo
import vacbo
import copy
import numpy as np
from matplotlib import pyplot as plt
from keep_default_optimizer import KeepDefaultOpt
from grid_search_optimizer import GridSearchOpt
from try_and_search_optimizer import SeqGridSearchOpt


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


def evaluate_one_fixed_control(fixed_param, discomfort_thr,
                               discomfort_weight=0.01,
                               optimizer_base_config=None,
                               optimization_config=None,
                               vars_to_fix=None):
    # try fixing one parameter
    no_opt_config = copy.deepcopy(optimizer_base_config)

    no_opt, no_opt_best_obj_list, no_opt_total_cost_list = get_vacbo_optimizer(
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


def evaluate_seq_control(fixed_param, discomfort_thr,
                         control_seq, discomfort_weight=0.01,
                         optimizer_base_config=None,
                         optimization_config=None,
                         vars_to_fix=None):
    # try fixing one parameter
    no_opt_config = copy.deepcopy(optimizer_base_config)

    no_opt, no_opt_best_obj_list, no_opt_total_cost_list = get_vacbo_optimizer(
        no_opt_config['problem_name'], 'no opt', no_opt_config,
        discomfort_thr=discomfort_thr, vars_to_fix=vars_to_fix,
        fixed_param=fixed_param, discomfort_weight=discomfort_weight)
    num_of_control = len(control_seq[:, 0])
    for _ in range(num_of_control):
        y_obj, constr_vals = no_opt.make_step(control_seq[_, :])
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
    return energy_per_day, ave_discomfort, cumulative_energy, \
        cumulative_discomfort


def evaluate_fixed_param(problem_name, fixed_param, num_evaluate_days,
                         init_points_id=0, discomfort_thr=1.0, vars_to_fix=[]):
    problem_config = util.get_config(
        problem_name,
        init_points_id=init_points_id,
        discomfort_thr=discomfort_thr,
        vars_to_fix=vars_to_fix
    )
    problem = vabo.optimization_problem.OptimizationProblem(problem_config)
    for day in range(num_evaluate_days):
        problem.sample_point(fixed_param)
    return problem


def evaluate_opt_params(opt, num_evaluate_days, problem_name, init_points_id=0,
                        discomfort_thr=1.0, vars_to_fix=[], top_k=5):
    evaluate_simulator_list = []
    feasible_points = opt.opt_problem.evaluated_feasible_points_list
    feasible_objs = opt.opt_problem.evaluated_feasible_objs_list
    sorted_feasible_points = [x for _, x in
                              sorted(zip(feasible_objs, feasible_points))]
    num_evaluated = 0
    for point in sorted_feasible_points:
        if num_evaluated > top_k:
            break
        num_evaluated += 1

        problem = evaluate_fixed_param(problem_name, point, num_evaluate_days,
                                       init_points_id, discomfort_thr,
                                       vars_to_fix)
        evaluate_simulator_list.append(problem.simulator)
    return evaluate_simulator_list


def get_vacbo_optimizer(problem_name, optimizer_type, optimizer_config,
                        init_points_id=0, vars_to_fix=None,
                        contextual_vars=None, fixed_param=None,
                        discomfort_weight=0.01, tune_var_scale='log',
                        tune_obj='energy', energy_thr=15.0,
                        problem_config=None):
    if problem_config is None:
        problem_config = util.get_config(
            problem_name, gp_kernel='Gaussian', init_points_id=init_points_id,
            vars_to_fix=vars_to_fix, contextual_vars=contextual_vars)
    print('Config got.')
    optimizer_config.update(problem_config)
    if fixed_param is not None:
        problem_config['init_safe_points'] = fixed_param
    problem = vacbo.ContextualOptimizationProblem(problem_config)
    if optimizer_type == 'safe_bo':
        opt = vacbo.safe_contextual_bo.SafeContextualBO(problem,
                                                        optimizer_config)
        total_cost_list = [opt.safe_bo.cumu_vio_cost]
    if optimizer_type == 'constrained_bo':
        opt = vacbo.constrained_contextual_bo.ConstrainedContextualBO(
            problem,
            optimizer_config)
        total_cost_list = [opt.constrained_bo.cumu_vio_cost]
    if optimizer_type == 'violation_aware_bo':
        opt = vacbo.violation_aware_contextual_bo.ViolationAwareContextualBO(
            problem, optimizer_config)
        total_cost_list = [opt.violation_aware_bo.cumu_vio_cost]
    if optimizer_type == 'lcb2bo':
        opt = vacbo.lcb2bo_contextual.LCB2ContextualBO(
            problem, optimizer_config)
        total_cost_list = [opt.lcb2_bo.cumu_vio_cost]

    if optimizer_type == 'pdcbo':
        opt = vacbo.pd_contextual_bo.PDContextualBO(
            problem, optimizer_config)
        total_cost_list = [opt.pdbo.cumu_vio_cost]
    if optimizer_type == 'no opt':
        opt = KeepDefaultOpt(problem, optimizer_config)
        total_cost_list = [opt.cumu_vio_cost]
    if optimizer_type == 'grid search':
        opt = GridSearchOpt(problem, optimizer_config)
        total_cost_list = [opt.cumu_vio_cost]

    if optimizer_type == 'seq grid search':
        opt = SeqGridSearchOpt(problem, optimizer_config)
        total_cost_list = [opt.cumu_vio_cost]

    return opt, total_cost_list, problem


def get_vabo_optimizer(problem_name, optimizer_type, optimizer_config,
                       init_points_id=0, discomfort_thr=1.0, vars_to_fix=[]):
    problem_config = util.get_config(
        problem_name,
        init_points_id=init_points_id,
        discomfort_thr=discomfort_thr,
        vars_to_fix=vars_to_fix
    )
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

    total_cost_list = [opt.cumu_vio_cost]
    return opt, best_obj_list, total_cost_list


def transfer_opt_result_dict(opt_result_dict, discft_list):
    opt_num_data_list = [
        len(opt_result_dict[i].opt_problem.simulator.history_dict.keys())
        for i in discft_list
    ]
    opt_discomfort_list = [
        opt_result_dict[i].opt_problem.simulator.cumulative_discomfort/(
            opt_num_data_list[i]/24.0
        )
        for i in discft_list
    ]
    opt_energy_list = [
        opt_result_dict[i].opt_problem.simulator.cumulative_energy/(
            opt_num_data_list[i]/(24.0 * 4)
        )
        for i in discft_list
    ]
    return opt_discomfort_list, opt_energy_list, opt_num_data_list
