import numpy as np
import simple_util
import matplotlib.pyplot as plt
from test_WO_util import get_optimizer, get_init_obj_constrs
from test_WO_util import get_safe_bo_result, get_constrained_bo_result
from test_WO_util import get_lcb2_result, get_pdbo_result
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

optimization_config = {
    'eval_budget': 200
}

safe_bo_config = {
    'noise_level': 0.0,
    'kernel_var': 0.1,
    'train_noise_level': 0.0,
    'problem_name': 'WO'
}


#violation_aware_opt.gp_obj.plot()

# compare cost of different methods
EPSILON=1e-4


total_eva_num = 100
vio_budgets_list = [0.0, 10.0, 20.0]

#vabo_
#safe_
def run_one_instance(x):
    print('Start running one instance!')
    global vio_budgets_list
    problem_name = 'WO'
    problem_config = simple_util.get_config(problem_name)
    try:
    #if True:
        print('Start running lcb2 optimizer!')
        lcb2_costs, lcb2_objs, lcb2_opt, lcb2_obj_traj, lcb2_constrs\
            = get_lcb2_result(problem_config)

        print('Stop running lcb2 optimizer!')
        safe_costs, safe_objs, safe_opt, safe_obj_traj, safe_constrs\
            = get_safe_bo_result(problem_config, plot=False)
        print('Start running con bo.')
        con_costs, con_objs, con_opt, con_obj_traj, con_constrs\
            = get_constrained_bo_result(problem_config, plot=False)
        pdbo_costs, pdbo_objs, pdbo_opt, pdbo_obj_traj, pdbo_constrs\
            = get_pdbo_result(problem_config)


        vabo_costs_tmp = []
        vabo_objs_tmp = []
        vabo_obj_traj_tmp = []
        vabo_constrs_tmp = []
        #for budget_id in range(len(vio_budgets_list)):
        #    budget = vio_budgets_list[budget_id]
        #    vabo_costs, vabo_objs, vabo_opt, vabo_obj_traj, vabo_constrs = \
        #        get_vabo_result(problem_config, plot=False, vio_budget=budget)
        #    vabo_costs_tmp.append(vabo_costs)
        #    vabo_objs_tmp.append(vabo_objs)
        #    vabo_obj_traj_tmp.append(vabo_obj_traj)
        #    vabo_constrs_tmp.append(vabo_constrs)
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
    problem_config = simple_util.get_config('WO')
    multi_results = []
    for _ in range(1):
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
                    np.array(safe_objs))
                safe_regret_lists.append(
                     - np.array(safe_obj_traj)
                )
                safe_constrs_lists.append(
                    np.array(safe_constrs)
                )

                con_bo_cost_lists.append(con_costs)
                con_bo_simple_regret_lists.append(
                    np.array(con_objs))
                con_bo_regret_lists.append(
                    np.array(con_obj_traj)
                )
                con_bo_constrs_lists.append(
                    np.array(con_constrs)
                )

                pdbo_cost_lists.append(pdbo_costs)
                pdbo_simple_regret_lists.append(
                    np.array(pdbo_objs)
                )
                pdbo_regret_lists.append(
                    np.array(pdbo_obj_traj)
                )
                pdbo_constrs_lists.append(
                    np.array(pdbo_constrs)
                )

                lcb2_cost_lists.append(lcb2_costs)
                lcb2_simple_regret_lists.append(
                    np.array(lcb2_objs))
                lcb2_regret_lists.append(
                    np.array(lcb2_obj_traj)
                )
                lcb2_constrs_lists.append(
                    np.array(lcb2_constrs)
                )

                for budget_id in range(len(vio_budgets_list)):
                    try:
                        vabo_cost_lists_set[budget_id].append(
                            vabo_costs_tmp[budget_id])
                        vabo_simple_regret_lists_set[budget_id].append(
                            np.array(vabo_objs_tmp[budget_id])
                        )
                        vabo_regret_lists_set[budget_id].append(
                            np.array(vabo_obj_traj_tmp[budget_id])
                        )
                        vabo_constrs_lists_set[budget_id].append(
                            np.array(vabo_constrs_tmp[budget_id])
                        )
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
        #for budget_id in range(len(vio_budgets_list)):
        #    vabo_ave_cost_arr_set.append((np.mean(np.array(vabo_cost_lists_set[budget_id]), axis=0), vio_budgets_list[budget_id]))
        #    vabo_ave_simple_regret_arr_set.append((np.mean(np.array(vabo_simple_regret_lists_set[budget_id]), axis=0), vio_budgets_list[budget_id]))
        #    vabo_ave_regret_arr_set.append((np.mean(np.array(vabo_regret_lists_set[budget_id]), axis=0), vio_budgets_list[budget_id]))

        np.savez('test_WO_result_with_pdbo', safe_ave_cost_arr, safe_ave_simple_regret_arr, con_ave_cost_arr, con_ave_simple_regret_arr, vabo_ave_cost_arr_set,
                vabo_ave_simple_regret_arr_set, safe_cost_lists, safe_simple_regret_lists, con_bo_cost_lists, con_bo_simple_regret_lists, vabo_cost_lists, vabo_simple_regret_lists,
            pdbo_cost_arr, pdbo_simple_regret_arr,
            safe_regret_lists, safe_constrs_lists, con_bo_regret_lists,
            con_bo_constrs_lists, pdbo_regret_lists, pdbo_constrs_lists,
            vabo_regret_lists_set, vabo_constrs_lists_set, lcb2_regret_lists,
            lcb2_constrs_lists)


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
