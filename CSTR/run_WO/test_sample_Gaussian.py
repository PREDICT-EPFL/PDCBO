import numpy as np
import safe_optimizer
import simple_util
import optimization_problem
import constrained_bo
import pdbo
import matplotlib.pyplot as plt
import violation_aware_bo
from lcb2 import LCB2
optimization_config = {
    'eval_budget': 20
}
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


# test SafeBO on the test function
def get_optimizer(optimizer_type, optimizer_config, problem_config):
    problem = optimization_problem.OptimizationProblem(problem_config)
    if optimizer_type == 'safe_bo':
        opt = safe_optimizer.SafeBO(problem, optimizer_config)
        best_obj_list = [-opt.best_obj]
    if optimizer_type == 'constrained_bo':
        opt = constrained_bo.ConstrainedBO(problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    if optimizer_type == 'violation_aware_bo':
        opt = violation_aware_bo.ViolationAwareBO(problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    if optimizer_type == 'pdbo':
        opt = pdbo.pd_bo.PDBO(problem, optimizer_config)
        best_obj_list = [opt.best_obj]
    if optimizer_type == 'lcb2':
        opt = LCB2(problem, optimizer_config)
        best_obj_list = [opt.best_obj]

    total_cost_list = [opt.cumu_vio_cost]
    return opt, best_obj_list, total_cost_list


def plot_sampled_data(config_list):
    legend_list = []
    i = 1
    for config in config_list:
        plt.plot(config['train_X'], config['obj'](config['train_X']))
        legend_list.append('sampled function '+str(i))
        i+=1
    all_zeros = np.zeros_like(config['train_X'])
    plt.plot(config['train_X'], all_zeros, linestyle='--')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim((-5, 5))
    plt.legend(legend_list)
    plt.savefig('./fig/GP_1_dim_sampled_funcs_new.pdf', format='pdf')


safe_bo_config = {
    'noise_level':0.0,
    'kernel_var':0.1,
    'train_noise_level': 0.0,
    'problem_name':'GP_sample_two_funcs'
}

def get_init_obj_constrs(opt):
    init_obj_val_arr, init_constr_val_arr = \
            opt.get_obj_constr_val(opt.x0_arr)
    init_obj_val_list = [init_obj_val_arr[0, 0]]
    init_constr_val_list = [init_constr_val_arr[0, :]]
    return init_obj_val_list, init_constr_val_list

def get_safe_bo_result(problem_config, plot=False):
    safe_opt, safe_bo_best_obj_list, safe_bo_total_cost_list = \
        get_optimizer('safe_bo', safe_bo_config, problem_config)

    init_obj_val_list, init_constr_val_list = get_init_obj_constrs(safe_opt)

    safe_opt_obj_list = init_obj_val_list
    safe_opt_constr_list = init_constr_val_list
    for _ in range(optimization_config['eval_budget']):
        y_obj, constr_vals = safe_opt.make_step()
        safe_bo_total_cost_list.append(safe_opt.cumu_vio_cost)
        safe_bo_best_obj_list.append(-safe_opt.best_obj)
        safe_opt_obj_list.append(y_obj)
        safe_opt_constr_list.append(constr_vals)
    if plot:
        safe_opt.plot()
        for i in range(safe_opt.opt_problem.num_constrs):
            plt.figure()
            plt.plot(np.array(safe_bo_total_cost_list)[:, i])
        plt.figure()
        plt.plot(safe_bo_best_obj_list)
    return safe_bo_total_cost_list, safe_bo_best_obj_list, safe_opt, \
        safe_opt_obj_list, safe_opt_constr_list

# test ConstrainedBO on the test function

def get_constrained_bo_result(problem_config, plot=False):
    constrained_opt, constrained_bo_best_obj_list, \
        constrained_bo_total_cost_list = get_optimizer(
            'constrained_bo', safe_bo_config, problem_config)

    init_obj_val_list, init_constr_val_list = get_init_obj_constrs(
        constrained_opt)
    constrained_opt_obj_list = init_obj_val_list
    constrained_opt_constr_list = init_constr_val_list
    for _ in range(optimization_config['eval_budget']):
    #if True:
        y_obj, constr_vals = constrained_opt.make_step()
        constrained_bo_total_cost_list.append(constrained_opt.cumu_vio_cost)
        constrained_bo_best_obj_list.append(constrained_opt.best_obj)
        constrained_opt_obj_list.append(y_obj)
        constrained_opt_constr_list.append(constr_vals)
    if plot:
        constrained_opt.plot()
        for i in range(constrained_opt.opt_problem.num_constrs):
            plt.figure()
            plt.plot(np.array(constrained_bo_total_cost_list)[:, i])
        plt.figure()
        plt.plot(constrained_bo_best_obj_list)
    return constrained_bo_total_cost_list, constrained_bo_best_obj_list, \
        constrained_opt, constrained_opt_obj_list, constrained_opt_constr_list

# test ViolationAwareBO on the test function

def get_vabo_result(problem_config, plot=False, vio_budget=5.0):
#plot=True
#vio_budget=5.0
#if True:
    violation_aware_bo_config = safe_bo_config.copy()
    violation_aware_bo_config.update({
        'total_vio_budgets': np.array([vio_budget]),
        'prob_eps': 1e-2,
        'beta_0': 10,
        'total_eval_num': optimization_config['eval_budget'],
        'normalize_input': False
        })
    violation_aware_opt, violation_aware_bo_best_obj_list, \
        violation_aware_bo_total_cost_list = get_optimizer(
            'violation_aware_bo', violation_aware_bo_config, problem_config)
    init_obj_val_list, init_constr_val_list = get_init_obj_constrs(
        violation_aware_opt)

    violation_aware_opt_obj_list = init_obj_val_list
    violation_aware_opt_constr_list = init_constr_val_list
    for _ in range(optimization_config['eval_budget']):
    #if True:
        #if True:
        y_obj, constr_vals = violation_aware_opt.make_step()
        violation_aware_bo_total_cost_list.append(
            violation_aware_opt.cumu_vio_cost)
        violation_aware_bo_best_obj_list.append(violation_aware_opt.best_obj)
        violation_aware_opt_obj_list.append(y_obj)
        violation_aware_opt_constr_list.append(constr_vals)
    if plot:
        violation_aware_opt.plot()
        for i in range(violation_aware_opt.opt_problem.num_constrs):
            plt.figure()
            plt.plot(np.array(violation_aware_bo_total_cost_list)[:, i])
        plt.figure()
        plt.plot(violation_aware_bo_best_obj_list)
    return violation_aware_bo_total_cost_list, \
        violation_aware_bo_best_obj_list, violation_aware_opt, \
        violation_aware_opt_obj_list, violation_aware_opt_constr_list


# test LCB2 on the test function
def get_lcb2_result(problem_config, plot=False):
    lcb2_config = safe_bo_config.copy()
    lcb2_config.update({
        'total_eval_num': optimization_config['eval_budget'],
        }
    )
    lcb2_opt, lcb2_best_obj_list, lcb2_total_cost_list = get_optimizer(
            'lcb2', lcb2_config, problem_config)
    lcb2_opt_obj_list = lcb2_opt.init_obj_val_list
    lcb2_opt_constr_list = lcb2_opt.init_constr_val_list
    for _ in range(optimization_config['eval_budget']):
    #if True:
        #if True:
        y_obj, constr_vals = lcb2_opt.make_step()
        lcb2_total_cost_list.append(
            lcb2_opt.cumu_vio_cost)
        lcb2_best_obj_list.append(lcb2_opt.best_obj)
        lcb2_opt_obj_list.append(y_obj)
        lcb2_opt_constr_list.append(constr_vals)
    if plot:
        lcb2_opt.plot()
        for i in range(lcb2_opt.opt_problem.num_constrs):
            plt.figure()
            plt.plot(np.array(lcb2_total_cost_list)[:, i])
        plt.figure()
        plt.plot(lcb2_best_obj_list)
    return lcb2_total_cost_list, \
        lcb2_best_obj_list, lcb2_opt, \
        lcb2_opt_obj_list, lcb2_opt_constr_list

def get_pdbo_result(problem_config, plot=False):
    pdbo_config = safe_bo_config.copy()
    pdbo_config.update({
        'beta_0': 3,
        'eta_0': 2,
        'total_eval_num': optimization_config['eval_budget'],
        'normalize_input': False
        })
    pdbo_opt, pdbo_best_obj_list, \
        pdbo_total_cost_list = get_optimizer(
            'pdbo', pdbo_config, problem_config)
    init_obj_val_list, init_constr_val_list = get_init_obj_constrs(
        pdbo_opt)

    pdbo_obj_list = init_obj_val_list
    pdbo_constr_val_list = init_constr_val_list
    for _ in range(optimization_config['eval_budget']):
    #if True:
        #if True:
        y_obj, constr_vals = pdbo_opt.make_step()
        pdbo_total_cost_list.append(
            pdbo_opt.cumu_vio_cost)
        pdbo_best_obj_list.append(pdbo_opt.best_obj)
        pdbo_obj_list.append(y_obj)
        pdbo_constr_val_list.append(constr_vals)
    if plot:
        pdbo_opt.plot()
        for i in range(pdbo_opt.opt_problem.num_constrs):
            plt.figure()
            plt.plot(np.array(pdbo_total_cost_list)[:, i])
        plt.figure()
        plt.plot(pdbo_best_obj_list)
    return pdbo_total_cost_list, pdbo_best_obj_list, pdbo_opt, \
        pdbo_obj_list, pdbo_constr_val_list

#violation_aware_opt.gp_obj.plot()

# compare cost of different methods
EPSILON=1e-4
def plot_results(
    safe_bo_total_cost_list, safe_bo_best_obj_list,
    constrained_bo_total_cost_list, constrained_bo_best_obj_list,
    violation_aware_bo_total_cost_list_set, violation_aware_bo_best_obj_list_set):
    plt.figure()
    plt.plot(np.array(safe_bo_total_cost_list)[:,  0], marker='*')
    plt.plot(np.array(constrained_bo_total_cost_list)[:, 0], marker='o')
    legends_list = ['Safe BO', 'Generic Constrained BO']

    for violation_aware_bo_total_cost_list, budget in violation_aware_bo_total_cost_list_set:
        plt.plot(np.array(violation_aware_bo_total_cost_list)[:, 0], marker='+')
        legends_list.append('Violation Aware BO '+str(budget))
        #print(violation_aware_bo_total_cost_list)

    #plt.plot(np.array([vio_budget]*(optimization_config['eval_budget']+1)), marker='v', color='r')
    plt.xlim((0, optimization_config['eval_budget']+1))
    #plt.ylim((0, 500))
    print(legends_list)
    plt.legend(legends_list)
    plt.xlabel('Step')
    plt.ylabel('Cumulative cost')
    plt.savefig('./fig/cost_inc.png', format='png')
    # compare convergence
    plt.figure()
    plt.plot(safe_bo_best_obj_list, marker='*')
    plt.plot(constrained_bo_best_obj_list, marker='o')

    for violation_aware_bo_best_obj_list, budget in violation_aware_bo_best_obj_list_set:
        plt.plot(violation_aware_bo_best_obj_list, marker='+')

    plt.xlim((0, optimization_config['eval_budget']+1))
    #plt.ylim((0, 500))
    #plt.xlim((0, 20))
    plt.legend(legends_list)
    plt.xlabel('step')
    plt.ylabel('Simple Regret')
    plt.savefig('./fig/best_obj.png', format='png')


#get_vabo_result(problem_config, plot=True, vio_budget=10.0)


# In[ ]:



total_eva_num = 100
vio_budgets_list = [0.0, 10.0, 20.0]

#vabo_
#safe_
def run_one_instance(x):
    global vio_budgets_list
    problem_name = 'GP_sample_two_funcs'
    problem_config = simple_util.get_config(problem_name)
    try:
    #if True:
        lcb2_costs, lcb2_objs, lcb2_opt, lcb2_obj_traj, lcb2_constrs\
            = get_lcb2_result(problem_config)

        safe_costs, safe_objs, safe_opt, safe_obj_traj, safe_constrs\
            = get_safe_bo_result(problem_config, plot=False)
        con_costs, con_objs, con_opt, con_obj_traj, con_constrs\
            = get_constrained_bo_result(problem_config, plot=False)
        pdbo_costs, pdbo_objs, pdbo_opt, pdbo_obj_traj, pdbo_constrs\
            = get_pdbo_result(problem_config)


        vabo_costs_tmp = []
        vabo_objs_tmp = []
        vabo_obj_traj_tmp = []
        vabo_constrs_tmp = []
        for budget_id in range(len(vio_budgets_list)):
            budget = vio_budgets_list[budget_id]
            vabo_costs, vabo_objs, vabo_opt, vabo_obj_traj, vabo_constrs = \
                get_vabo_result(problem_config, plot=False, vio_budget=budget)
            vabo_costs_tmp.append(vabo_costs)
            vabo_objs_tmp.append(vabo_objs)
            vabo_obj_traj_tmp.append(vabo_obj_traj)
            vabo_constrs_tmp.append(vabo_constrs)
            # print(vabo_costs, budget_id)
    except Exception as e:
    #else:
        print(e)
        return None, None, None, None, None, None, None, None, None, None,\
            None, None, None, None, None, None, None, None, None, None, None
    return safe_costs, safe_objs, con_costs, con_objs, vabo_costs_tmp, \
        vabo_objs_tmp, pdbo_costs, pdbo_objs, safe_obj_traj, safe_constrs, \
        con_obj_traj, con_constrs, pdbo_obj_traj, pdbo_constrs, \
        vabo_obj_traj_tmp, vabo_constrs_tmp, lcb2_costs, lcb2_objs, lcb2_opt, \
        lcb2_obj_traj, lcb2_constrs\



if __name__ == '__main__':
    problem_config = simple_util.get_config('GP_sample_two_funcs')
    multi_results = []
    for _ in range(50):
        safe_costs, safe_objs, con_costs, con_objs, vabo_costs_tmp, \
            vabo_objs_tmp, pdbo_costs, pdbo_objs, safe_obj_traj, safe_constrs, \
            con_obj_traj, con_constrs, pdbo_obj_traj, pdbo_constrs, \
            vabo_obj_traj_tmp, vabo_constrs_tmp, lcb2_costs, lcb2_objs, lcb2_opt, \
            lcb2_obj_traj, lcb2_constrs = run_one_instance(0)
        multi_results.append((safe_costs, safe_objs, con_costs, con_objs,
                         vabo_costs_tmp, vabo_objs_tmp, pdbo_costs, pdbo_objs,
                         safe_obj_traj, safe_constrs, con_obj_traj,
                          con_constrs, pdbo_obj_traj, pdbo_constrs,
                          vabo_obj_traj_tmp, vabo_constrs_tmp, lcb2_costs, \
                          lcb2_objs, lcb2_opt, lcb2_obj_traj, lcb2_constrs))

        safe_cost_lists = []
        safe_simple_regret_lists = []
        safe_regret_lists = []
        safe_constrs_lists = []

        con_bo_cost_lists = []
        con_bo_simple_regret_lists = []
        con_bo_regret_lists = []
        con_bo_constrs_lists = []

        vabo_cost_lists = []
        vabo_simple_regret_lists = []
        vabo_regret_lists = []
        vabo_constrs_lists = []

        vabo_cost_lists_set = [[] for _ in range(len(vio_budgets_list))]
        vabo_simple_regret_lists_set = [[] for i in range(len(vio_budgets_list))]
        vabo_regret_lists_set = [[] for _ in range(len(vio_budgets_list))]
        vabo_constrs_lists_set = [[] for _ in range(len(vio_budgets_list))]

        pdbo_cost_lists = []
        pdbo_simple_regret_lists = []
        pdbo_regret_lists = []
        pdbo_constrs_lists = []

        lcb2_cost_lists = []
        lcb2_simple_regret_lists = []
        lcb2_regret_lists = []
        lcb2_constrs_lists = []

        for safe_costs, safe_objs, con_costs, con_objs, vabo_costs_tmp, vabo_objs_tmp,\
            pdbo_costs, pdbo_objs, safe_obj_traj, safe_constrs, con_obj_traj, \
            con_constrs, pdbo_obj_traj, pdbo_constrs, vabo_obj_traj_tmp, \
            vabo_constrs_tmp, lcb2_costs, lcb2_objs, lcb2_opt, lcb2_obj_traj, \
            lcb2_constrs in multi_results:
            if safe_costs is not None:
                safe_cost_lists.append(safe_costs)
                safe_simple_regret_lists.append(
                    np.array(safe_objs)-problem_config['f_min'])
                safe_regret_lists.append(
                    -problem_config['f_min'] - np.array(safe_obj_traj)
                )
                safe_constrs_lists.append(
                    np.array(safe_constrs)
                )

                con_bo_cost_lists.append(con_costs)
                con_bo_simple_regret_lists.append(
                    np.array(con_objs)-problem_config['f_min'])
                con_bo_regret_lists.append(
                    np.array(con_obj_traj)-problem_config['f_min']
                )
                con_bo_constrs_lists.append(
                    np.array(con_constrs)
                )

                pdbo_cost_lists.append(pdbo_costs)
                pdbo_simple_regret_lists.append(
                    np.array(pdbo_objs)-problem_config['f_min'])
                pdbo_regret_lists.append(
                    np.array(pdbo_obj_traj)-problem_config['f_min']
                )
                pdbo_constrs_lists.append(
                    np.array(pdbo_constrs)
                )

                lcb2_cost_lists.append(lcb2_costs)
                lcb2_simple_regret_lists.append(
                    np.array(lcb2_objs)-problem_config['f_min'])
                lcb2_regret_lists.append(
                    np.array(lcb2_obj_traj)-problem_config['f_min']
                )
                lcb2_constrs_lists.append(
                    np.array(lcb2_constrs)
                )

                for budget_id in range(len(vio_budgets_list)):
                    try:
                        vabo_cost_lists_set[budget_id].append(
                            vabo_costs_tmp[budget_id])
                        vabo_simple_regret_lists_set[budget_id].append(
                            np.array(vabo_objs_tmp[budget_id])-problem_config['f_min'])
                        vabo_regret_lists_set[budget_id].append(
                            np.array(vabo_obj_traj_tmp[budget_id])-problem_config['f_min'])
                        vabo_constrs_lists_set[budget_id].append(
                            np.array(vabo_constrs_tmp[budget_id])-problem_config['f_min'])
                    except Exception as e:
                        print(f'Exception {e}.')
                        continue

        safe_ave_cost_arr = np.mean(np.array(safe_cost_lists), axis=0)
        safe_ave_simple_regret_arr = np.mean(np.array(safe_simple_regret_lists), axis=0)
        safe_ave_regret_arr = np.mean(np.array(safe_regret_lists), axis=0)

        con_ave_cost_arr = np.mean(np.array(con_bo_cost_lists), axis=0)
        con_ave_simple_regret_arr = np.mean(np.array(con_bo_simple_regret_lists), axis=0)
        con_ave_regret_arr = np.mean(np.array(con_bo_regret_lists), axis=0)

        pdbo_cost_arr = np.mean(np.array(pdbo_cost_lists), axis=0)
        pdbo_simple_regret_arr = np.mean(np.array(pdbo_simple_regret_lists), axis=0)
        pdbo_regret_arr = np.mean(np.array(pdbo_regret_lists), axis=0)

        lcb2_cost_arr = np.mean(np.array(lcb2_cost_lists), axis=0)
        lcb2_simple_regret_arr = np.mean(np.array(lcb2_simple_regret_lists), axis=0)
        lcb2_regret_arr = np.mean(np.array(lcb2_regret_lists), axis=0)

        vabo_ave_cost_arr_set = []
        vabo_ave_simple_regret_arr_set = []
        vabo_ave_regret_arr_set = []
        for budget_id in range(len(vio_budgets_list)):
            vabo_ave_cost_arr_set.append((np.mean(np.array(vabo_cost_lists_set[budget_id]), axis=0), vio_budgets_list[budget_id]))
            vabo_ave_simple_regret_arr_set.append((np.mean(np.array(vabo_simple_regret_lists_set[budget_id]), axis=0), vio_budgets_list[budget_id]))
            vabo_ave_regret_arr_set.append((np.mean(np.array(vabo_regret_lists_set[budget_id]), axis=0), vio_budgets_list[budget_id]))

        np.savez('test_GP_sample_result_with_pdbo', safe_ave_cost_arr, safe_ave_simple_regret_arr, con_ave_cost_arr, con_ave_simple_regret_arr, vabo_ave_cost_arr_set,
                vabo_ave_simple_regret_arr_set, safe_cost_lists, safe_simple_regret_lists, con_bo_cost_lists, con_bo_simple_regret_lists, vabo_cost_lists, vabo_simple_regret_lists,
            pdbo_cost_arr, pdbo_simple_regret_arr,
            safe_regret_lists, safe_constrs_lists, con_bo_regret_lists,
            con_bo_constrs_lists, pdbo_regret_lists, pdbo_constrs_lists,
            vabo_regret_lists_set, vabo_constrs_lists_set, lcb2_regret_lists,
            lcb2_constrs_lists)


    plot_results(safe_ave_cost_arr, safe_ave_simple_regret_arr, con_ave_cost_arr, con_ave_simple_regret_arr, vabo_ave_cost_arr_set,
            vabo_ave_simple_regret_arr_set)


#safe_ave_cost_arr


def plot_cost_SR_scatter(cost_lists, SR_lists, fig_name=None):
    cost_arr = np.array(cost_lists)
    num_tra, num_eval, num_constr = cost_arr.shape

    SR_arr = np.array(SR_lists)

    for i in range(num_constr):
        plt.figure()
        plt.scatter(cost_arr[:,-1,i], SR_arr[:,-1])
        plt.xlabel('Violation cost '+str(i+1))
        plt.ylabel('Simple regret')
        plt.title(fig_name)

    if fig_name is not None:
        plt.savefig('./fig/'+fig_name, format='png')


#plot_cost_SR_scatter(safe_cost_lists, safe_simple_regret_lists, 'Safe BO cost-simple regret')
#plot_cost_SR_scatter(con_bo_cost_lists, con_bo_simple_regret_lists, 'Constrained BO cost-simple regret')
#plot_cost_SR_scatter(vabo_cost_lists_set[0], vabo_simple_regret_lists_set[0], 'Violation-aware BO cost-simple regret 0.0')
#plot_cost_SR_scatter(vabo_cost_lists_set[1], vabo_simple_regret_lists_set[1], 'Violation-aware BO cost-simple regret 10.0')
#plot_cost_SR_scatter(vabo_cost_lists_set[2], vabo_simple_regret_lists_set[2], 'Violation-aware BO cost-simple regret 20.0')

#import merl_model.steady_state_analyze as MERL_model
