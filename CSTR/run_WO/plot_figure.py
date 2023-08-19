def read_simulation_data(npz_file):
    merl_simulator_opt_data = np.load(npz_file, allow_pickle=True)
    data = merl_simulator_opt_data['arr_0']
    total_cost_list = data[0]
    best_obj_list = data[1]
    opt_X = data[2]
    opt_points = data[3]
    opt_contrs = data[4]
    opt_objs = data[5]
    return total_cost_list, best_obj_list, opt_X, opt_points, opt_contrs, opt_objs

safe_bo_total_cost_list, safe_bo_best_obj_list, safe_opt_X, safe_opt_points, safe_opt_contrs, safe_opt_objs = read_simulation_data('merl_simulator_safe_opt.npz')#'./result/merl_simulator_safe_opt_beta_2_random_id_2_Td_58.npz')#'merl_simulator_safe_opt.npz')

constrained_bo_total_cost_list, constrained_bo_best_obj_list, constrained_opt_X, constrained_opt_points, constrained_opt_contrs, constrained_opt_objs = read_simulation_data('merl_simulator_constrained_opt.npz')#'./result/merl_simulator_constrained_opt_random_id_2_Td_58.npz')#'merl_simulator_constrained_opt.npz')

violation_aware_bo_total_cost_list, violation_aware_bo_best_obj_list, violation_aware_opt_X, violation_aware_opt_points, violation_aware_opt_contrs, violation_aware_opt_objs = read_simulation_data('merl_simulator_vabo_prob_0.01_budget_10.0_beta_1.npz')#'./result/merl_simulator_vabo_prob_0.01_budget_10.0_beta0_1_random_id_2_Td_58.npz')#'merl_simulator_vabo_prob_0.01_budget_10.0_beta_1.npz')

violation_aware_bo_total_cost_list_1, violation_aware_bo_best_obj_list_1, violation_aware_opt_X_1, violation_aware_opt_points_1, violation_aware_opt_contrs_1, violation_aware_opt_objs_1 = read_simulation_data('merl_simulator_vabo_prob_0.04_budget_0.0.npz')#'./result/merl_simulator_vabo_prob_0.01_budget_0.0_beta0_1_random_id_2_Td_58.npz')#'merl_simulator_vabo_prob_0.04_budget_0.0.npz')

# compare cost of different methods
vabo_1_len = 2
eval_num = 20
vio_budget = 10.0
T_d_thr = 331.15
T_e_thr = 278.15
font_size = 12
font = {'size'   : font_size}
import matplotlib
matplotlib.rc('font', **font)

from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator

ax = plt.figure().gca()

plt.plot(np.array(safe_bo_total_cost_list)[:,  0], marker='*')
plt.plot(np.array(constrained_bo_total_cost_list)[:, 0], marker='o')
plt.plot(np.array(violation_aware_bo_total_cost_list)[:, 0], marker='+')
plt.plot(np.array(violation_aware_bo_total_cost_list_1)[:, 0], marker='+')
plt.plot(np.array([10.0]*(optimization_config['eval_budget']+1)), linestyle='--', color='r')
plt.xlim((0, optimization_config['eval_budget']+1))
plt.ylim((0, 250))
plt.xlim(0, eval_num)
plt.legend(['Safe BO', 'Generic Constrained BO', 'Violation Aware BO $'+str(vio_budget)+'$', 'Violation Aware BO $0.0$', 'Violation cost budget $'+str(vio_budget)+'$'])
plt.xlabel('Step')
plt.ylabel('Cumulative cost~($([T_\mathrm{d}-\hat T_\mathrm{d}]^+)^2$)')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
#plt.savefig('./fig/cost_inc_merl_simulator.pdf', format='pdf')


# compare another cost
ax = plt.figure().gca()

plt.plot(np.array(safe_bo_total_cost_list)[:,  1], marker='*')
plt.plot(np.array(constrained_bo_total_cost_list)[:, 1], marker='o')
plt.plot(np.array(violation_aware_bo_total_cost_list)[:, 1], marker='+')
plt.plot(np.array(violation_aware_bo_total_cost_list_1)[:, 1], marker='+')
plt.plot(np.array([50.0]*(optimization_config['eval_budget']+1)), linestyle='--', color='r')
plt.xlim((0, optimization_config['eval_budget']+1))
plt.ylim((0, 250))
plt.xlim(0, eval_num)
plt.legend(['Safe BO', 'Generic Constrained BO', 'Violation Aware BO $'+str(vio_budget)+'$', 'Violation Aware BO $0.0$', 'Violation cost budget $'+str(vio_budget)+'$'])
plt.xlabel('Step')
plt.ylabel('Cumulative cost~($([\check T_\mathrm{e}-T_\mathrm{e}]^+)^2$)')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
#plt.savefig('./fig/cost_inc_constr_2_merl_simulator.pdf', format='pdf')


# compare constraints
ax = plt.figure().gca()

plt.plot(np.array(safe_opt_contrs[-eval_num:]).squeeze()[:, 0]+T_d_thr, marker='*')
plt.plot(np.array(constrained_opt_contrs[-eval_num:]).squeeze()[:, 0]+T_d_thr, marker='o')
plt.plot(np.array(violation_aware_opt_contrs[-eval_num:]).squeeze()[:, 0]+T_d_thr, marker='+')
plt.plot(np.array(violation_aware_opt_contrs_1[vabo_1_len:]).squeeze()[:, 0]+T_d_thr, marker='+')
plt.plot(np.array([0]*(optimization_config['eval_budget']+1))+T_d_thr, linestyle='--', color='r')
plt.xlim((0, optimization_config['eval_budget']+1))
#plt.ylim((0, 250))
plt.xlim(0, eval_num)
plt.legend(['Safe BO', 'Generic Constrained BO', 'Violation Aware BO $'+str(vio_budget)+'$', 'Violation Aware BO $0.0$', '$\hat T_\mathrm{d}$'])
plt.xlabel('Step')
plt.ylabel('$T_\mathrm{d}$ (unit: $K$)')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
#plt.savefig('./fig/constr_1_merl_simulator.pdf', format='pdf')

# compare constraints
ax = plt.figure().gca()

plt.plot(T_e_thr-np.array(safe_opt_contrs[-eval_num:]).squeeze()[:, 1], marker='*')
plt.plot(T_e_thr-np.array(constrained_opt_contrs[-eval_num:]).squeeze()[:, 1], marker='o')
plt.plot(T_e_thr-np.array(violation_aware_opt_contrs[-eval_num:]).squeeze()[:, 1], marker='+')
plt.plot(T_e_thr-np.array(violation_aware_opt_contrs_1[vabo_1_len:]).squeeze()[:, 1], marker='+')
plt.plot(T_e_thr-np.array([0]*(optimization_config['eval_budget']+1)), linestyle='--', color='r')
plt.xlim((0, optimization_config['eval_budget']+1))
#plt.ylim((0, 250))
plt.xlim(0, eval_num)
plt.legend(['Safe BO', 'Generic Constrained BO', 'Violation Aware BO $'+str(vio_budget)+'$', 'Violation Aware BO $0.0$', '$\check T_\mathrm{e}$'])
plt.xlabel('Step')
plt.ylabel('$T_\mathrm{e}$ (unit: $K$)')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
#plt.savefig('./fig/constr_2_merl_simulator.pdf', format='pdf')

# compare objective
ax = plt.figure().gca()
plt.plot(np.array(safe_opt_objs[-eval_num:]).squeeze(), marker='*')
plt.plot(np.array(constrained_opt_objs[-eval_num:]).squeeze(), marker='o')
plt.plot(np.array(violation_aware_opt_objs[-eval_num:]).squeeze(), marker='+')
plt.plot(np.array(violation_aware_opt_objs_1[vabo_1_len:]).squeeze(), marker='+')
#plt.plot(np.array([0]*(optimization_config['eval_budget']+1)), linestyle='--', color='r')
plt.xlim((0, optimization_config['eval_budget']+1))
#plt.ylim((0, 250))
plt.xlim(0, eval_num)
plt.legend(['Safe BO', 'Generic Constrained BO', 'Violation Aware BO $'+str(vio_budget)+'$', 'Violation Aware BO $0.0$'])
plt.xlabel('Step')
plt.ylabel('Power (unit: W)')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
#plt.savefig('./fig/objs_merl_simulator.pdf', format='pdf')

# compare convergence
#minimum, minimizer = violation_aware_opt.opt_problem.get_minimum()
ax = plt.figure().gca()
shift = 500
plt.plot(np.maximum(np.array(safe_bo_best_obj_list)+shift, 1e-4), marker='*')
plt.plot(np.maximum(np.array(constrained_bo_best_obj_list)+shift, 1e-4), marker='o')
plt.plot(np.maximum(np.array(violation_aware_bo_best_obj_list)+shift, 1e-4), marker='+')
plt.plot(np.maximum(np.array(violation_aware_bo_best_obj_list_1)+shift, 1e-4), marker='+')
plt.xlim((0, eval_num))
plt.legend(['Safe BO', 'Generic Constrained BO', 'Violation Aware BO $'+str(vio_budget)+'$', 'Violation Aware BO $0.0$'])
plt.xlabel('Step')
plt.ylabel('Best power function (unit:W)')
#ax.ticklabel_format(useOffset=False)
#plt.ticklabel_format(style='plain', axis='y')
#import matplotlib
#ax.get_yaxis().set_major_formatter(
#    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
#plt.savefig('./fig/best_obj_merl_simulator.pdf', format='pdf')