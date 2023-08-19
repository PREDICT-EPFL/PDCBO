import numpy as np
import simple_util
import datetime
from test_WO_util import get_safe_bo_result, get_constrained_bo_result
from test_WO_util import get_lcb2_result, get_pdbo_result, get_epbo_result
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

base_opt_config = {
    'noise_level': 0.0,
    'kernel_var': 0.1,
    'train_noise_level': 0.0,
    'problem_name': 'WO',
    'eval_budget': 100,
    'num_run': 1
}

epbo_config = base_opt_config.copy()
epbo_config.update({
        'total_eval_num': base_opt_config['eval_budget'],
        'penalty': 1.0
        }
    )

epbo_penalty_list = [0.2, 1.0, 3.0] # [1.0, 10.0, 100.0] #[0.0, 1e-7, 1e-5] #0.0001, 0.001, 0.01]


def run_one_instance(base_opt_config):
    print('Start running one instance!')
    problem_config = simple_util.get_config(
        problem_name=base_opt_config['problem_name'])

    try:
        print('Start running lcb2 optimizer!')
        lcb2_costs, lcb2_objs, lcb2_opt, lcb2_obj_traj, lcb2_constrs\
            = get_lcb2_result(problem_config, base_opt_config)

        print('Stop running lcb2 optimizer!')

        base_opt_config['penalty'] = epbo_penalty_list[0]
        epbo_costs_0, epbo_objs_0, epbo_opt_0, epbo_obj_traj_0, epbo_constrs_0\
            = get_epbo_result(problem_config, base_opt_config)

        base_opt_config['penalty'] = epbo_penalty_list[1]
        epbo_costs_1, epbo_objs_1, epbo_opt_1, epbo_obj_traj_1, epbo_constrs_1\
            = get_epbo_result(problem_config, base_opt_config)

        base_opt_config['penalty'] = epbo_penalty_list[2]
        epbo_costs_2, epbo_objs_2, epbo_opt_2, epbo_obj_traj_2, epbo_constrs_2\
            = get_epbo_result(problem_config, base_opt_config)

        if problem_config['init_safe']:
            safe_costs, safe_objs, safe_opt, safe_obj_traj, safe_constrs\
                = get_safe_bo_result(problem_config, base_opt_config)
        else:
            safe_costs, safe_objs, safe_opt, safe_obj_traj, safe_constrs\
                = [[0]], [[0]], [[0]], [[0]], [[0]]
        con_costs, con_objs, con_opt, con_obj_traj, con_constrs\
            = get_constrained_bo_result(problem_config, base_opt_config)
        pdbo_costs, pdbo_objs, pdbo_opt, pdbo_obj_traj, pdbo_constrs\
            = get_pdbo_result(problem_config, base_opt_config)
    except Exception as e:
        print(e)
        return None, None, None, None, None, None, None, None, None, None,\
            None, None, None, None, None, None, None
    return safe_costs, safe_objs, con_costs, con_objs, pdbo_costs, \
        pdbo_objs, safe_obj_traj, safe_constrs, \
        con_obj_traj, con_constrs, pdbo_obj_traj, pdbo_constrs, \
        lcb2_costs, lcb2_objs, lcb2_opt, \
        lcb2_obj_traj, lcb2_constrs, epbo_costs_0, epbo_objs_0, epbo_opt_0, \
        epbo_obj_traj_0, epbo_constrs_0, epbo_costs_1, epbo_objs_1, \
        epbo_opt_1, epbo_obj_traj_1, epbo_constrs_1, epbo_costs_2, \
        epbo_objs_2, epbo_opt_2, epbo_obj_traj_2, epbo_constrs_2, problem_config

if __name__ == '__main__':
    num_run = base_opt_config['num_run']
    multi_results = []
    for _ in range(30):
        safe_costs, safe_objs, con_costs, con_objs, \
            pdbo_costs, pdbo_objs, safe_obj_traj, safe_constrs, \
            con_obj_traj, con_constrs, pdbo_obj_traj, pdbo_constrs, \
            lcb2_costs, lcb2_objs, lcb2_opt, lcb2_obj_traj, lcb2_constrs, \
            epbo_costs_0, epbo_objs_0, epbo_opt_0, \
        epbo_obj_traj_0, epbo_constrs_0, epbo_costs_1, epbo_objs_1, \
        epbo_opt_1, epbo_obj_traj_1, epbo_constrs_1, epbo_costs_2, \
        epbo_objs_2, epbo_opt_2, epbo_obj_traj_2, epbo_constrs_2, problem_config \
            = run_one_instance(base_opt_config)
        multi_results.append((safe_costs, safe_objs, con_costs, con_objs,
                              pdbo_costs, pdbo_objs, safe_obj_traj,
                              safe_constrs, con_obj_traj, con_constrs,
                              pdbo_obj_traj, pdbo_constrs, lcb2_costs,
                              lcb2_objs, lcb2_opt, lcb2_obj_traj, lcb2_constrs,
            epbo_costs_0, epbo_objs_0, epbo_opt_0, \
        epbo_obj_traj_0, epbo_constrs_0, epbo_costs_1, epbo_objs_1, \
        epbo_opt_1, epbo_obj_traj_1, epbo_constrs_1, epbo_costs_2, \
        epbo_objs_2, epbo_opt_2, epbo_obj_traj_2, epbo_constrs_2,
                              problem_config
                              ))

        safe_cost_lists = []
        safe_simple_regret_lists = []
        safe_regret_lists = []
        safe_constrs_lists = []

        con_bo_cost_lists = []
        con_bo_simple_regret_lists = []
        con_bo_regret_lists = []
        con_bo_constrs_lists = []

        pdbo_cost_lists = []
        pdbo_simple_regret_lists = []
        pdbo_regret_lists = []
        pdbo_constrs_lists = []

        lcb2_cost_lists = []
        lcb2_simple_regret_lists = []
        lcb2_regret_lists = []
        lcb2_constrs_lists = []

        epbo_regret_lists_0 = []
        epbo_constrs_lists_0 = []

        epbo_regret_lists_1 = []
        epbo_constrs_lists_1 = []

        epbo_regret_lists_2 = []
        epbo_constrs_lists_2 = []

        for safe_costs, safe_objs, con_costs, con_objs, \
            pdbo_costs, pdbo_objs, safe_obj_traj, safe_constrs, con_obj_traj, \
            con_constrs, pdbo_obj_traj, pdbo_constrs, lcb2_costs, lcb2_objs, \
            lcb2_opt, lcb2_obj_traj, lcb2_constrs, epbo_costs_0, epbo_objs_0, epbo_opt_0, \
        epbo_obj_traj_0, epbo_constrs_0, epbo_costs_1, epbo_objs_1, \
        epbo_opt_1, epbo_obj_traj_1, epbo_constrs_1, epbo_costs_2, \
        epbo_objs_2, epbo_opt_2, epbo_obj_traj_2, epbo_constrs_2, \
            problem_config in multi_results:

            if safe_costs is not None:
                safe_cost_lists.append(safe_costs)
                safe_simple_regret_lists.append(
                    np.array(safe_objs))
                safe_regret_lists.append(
                     - np.array(safe_obj_traj) - problem_config['f_min']
                )
                safe_constrs_lists.append(
                    np.array(safe_constrs)
                )

                con_bo_cost_lists.append(con_costs)
                con_bo_simple_regret_lists.append(
                    np.array(con_objs))
                con_bo_regret_lists.append(
                    np.array(con_obj_traj) - problem_config['f_min']
                )
                con_bo_constrs_lists.append(
                    np.array(con_constrs)
                )

                pdbo_cost_lists.append(pdbo_costs)
                pdbo_simple_regret_lists.append(
                    np.array(pdbo_objs)
                )
                pdbo_regret_lists.append(
                    np.array(pdbo_obj_traj) - problem_config['f_min']
                )
                pdbo_constrs_lists.append(
                    np.array(pdbo_constrs)
                )

                lcb2_cost_lists.append(lcb2_costs)
                lcb2_simple_regret_lists.append(
                    np.array(lcb2_objs))
                lcb2_regret_lists.append(
                    np.array(lcb2_obj_traj) - problem_config['f_min']
                )
                lcb2_constrs_lists.append(
                    np.array(lcb2_constrs)
                )

                epbo_regret_lists_0.append(
                    np.array(epbo_obj_traj_0) - problem_config['f_min']
                )
                epbo_constrs_lists_0.append(
                    np.array(epbo_constrs_0)
                )

                epbo_regret_lists_1.append(
                    np.array(epbo_obj_traj_1) - problem_config['f_min']
                )
                epbo_constrs_lists_1.append(
                    np.array(epbo_constrs_1)
                )

                epbo_regret_lists_2.append(
                    np.array(epbo_obj_traj_2) - problem_config['f_min']
                )
                epbo_constrs_lists_2.append(
                    np.array(epbo_constrs_2)
                )

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


now_time_str = datetime.datetime.now().strftime(
        "%H_%M_%S-%b_%d_%Y")
np.savez(f'test_WO_epbo_{problem_config["init_safe"]}_{epbo_penalty_list[0]}_{epbo_penalty_list[1]}_{epbo_penalty_list[2]}_{now_time_str}',
         safe_ave_cost_arr, safe_ave_simple_regret_arr,
         con_ave_cost_arr, con_ave_simple_regret_arr, safe_cost_lists,
         safe_simple_regret_lists, con_bo_cost_lists,
         con_bo_simple_regret_lists, pdbo_cost_arr, pdbo_simple_regret_arr,
         safe_regret_lists, safe_constrs_lists, con_bo_regret_lists,
         con_bo_constrs_lists, pdbo_regret_lists, pdbo_constrs_lists,
         lcb2_regret_lists, lcb2_constrs_lists, epbo_regret_lists_0, \
         epbo_constrs_lists_0, epbo_regret_lists_1, epbo_constrs_lists_1,
         epbo_regret_lists_2, epbo_constrs_lists_2
         )
