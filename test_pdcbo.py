#!/usr/bin/env python
# coding: utf-8

# In[34]:


import util
import vabo
import vacbo
import copy
import numpy as np
from matplotlib import pyplot as plt
from keep_default_optimizer import KeepDefaultOpt
from grid_search_optimizer import GridSearchOpt
import os
from tune_util import get_vacbo_optimizer
#from 
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Configuration

# In[35]:


# parameter configurations to enumerate
discomfort_thr_list = list(range(5, 50, 10))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
optimization_config = {
    'eval_budget': 300
}

optimizer_base_config = {
    'noise_level': [0.004, 0.2, 0.2],
    'kernel_var': 0.1,
    'train_noise_level': 1.0,
    'problem_name': 'SinglePIRoomEvaluator',
    'normalize_input': False
}

VARS_TO_FIX = ['high_off_time', 'low_setpoint',
               'control_setpoint']
CONTEXTUAL_VARS = ['Q_irr', 'T_out', 'T_init']


tune_var_scale = 'log'
save_name_append = f'_{tune_var_scale}_with_context'
budget = 10.0
discomfort_thr = 15.0
discomfort_weight = 0.0

discomfort_thr_to_ave_energy_pdcbo = dict()
discomfort_thr_to_ave_discomfort_pdcbo = dict()
discomfort_thr_to_energy_pdcbo = dict()
discomfort_thr_to_discomfort_pdcbo = dict()

discomfort_thr_to_ave_energy_cbo = dict()
discomfort_thr_to_ave_discomfort_cbo = dict()
discomfort_thr_to_energy_cbo = dict()
discomfort_thr_to_discomfort_cbo = dict()

discomfort_thr_to_ave_energy_safe_bo = dict()
discomfort_thr_to_ave_discomfort_safe_bo = dict()
discomfort_thr_to_energy_safe_bo = dict()
discomfort_thr_to_discomfort_safe_bo = dict()

discomfort_thr_to_ave_energy_no_opt = dict()
discomfort_thr_to_ave_discomfort_no_opt = dict()
discomfort_thr_to_energy_no_opt = dict()
discomfort_thr_to_discomfort_no_opt = dict()

TUNE_OBJ = 'discomfort'
ENERGY_THR = 0.021


# # PDCBO

# In[39]:


pdcbo_config = copy.deepcopy(optimizer_base_config)
pdcbo_config.update({
        'eta_0': 10.0,
        'eta_func': lambda t: 5.0, 
        'total_eval_num': optimization_config['eval_budget'],
        'init_dual': 10.0,
        'lcb_coef': lambda t: 2.0 
    })

optimizer_type = 'pdcbo'
pdc_opt, opt_total_cost_list, opt_problem = get_vacbo_optimizer(
                    pdcbo_config['problem_name'], 'pdcbo', pdcbo_config,
                    discomfort_thr=discomfort_thr, vars_to_fix=VARS_TO_FIX,
                    contextual_vars=CONTEXTUAL_VARS,
                    discomfort_weight=discomfort_weight, tune_obj=TUNE_OBJ, energy_thr=ENERGY_THR)
opt_obj_list = []
constraints_list = []
energy_list = []
discomfort_list = []
average_energy_list = []
average_discomfort_list = []

if True:
#for _ in range(optimization_config['eval_budget']):
    context_vars = opt_problem.get_context(opt_problem.simulator)
    y_obj, constr_vals = pdc_opt.make_step(context_vars)
    if optimizer_type == 'safe_bo':
        new_cumu_cost = pdc_opt.safe_bo.cumu_vio_cost
    if optimizer_type == 'constrained_bo':
        new_cumu_cost = pdc_opt.constrained_bo.cumu_vio_cost
    if optimizer_type == 'violation_aware_bo':
        new_cumu_cost = pdc_opt.violation_aware_bo.cumu_vio_cost
    if optimizer_type == 'pdcbo':
        new_cumu_cost = pdc_opt.pdbo.cumu_vio_cost
    if optimizer_type == 'no opt':
        new_cumu_cost = pdc_opt.cumu_vio_cost
    if optimizer_type == 'grid search':
        new_cumu_cost = pdc_opt.cumu_vio_cost

    opt_total_cost_list.append(new_cumu_cost)
    opt_obj_list.append(y_obj)
    constraints_list.append(constr_vals)
    energy, discomfort = opt_problem.simulator.get_recent_energy_discomfort_per_day()
    
    

    energy_list.append(energy)
    discomfort_list.append(discomfort)
    print_log = True
    if print_log:
        print(f"In step {_}, with discomfort threshold " +
                f"{discomfort_thr} and discomfort weight " +
                f"{discomfort_weight}, we get energy {energy}" +
                f" and discomfort {discomfort}, with the point "
                + f" {opt_problem.evaluated_points_list[-1]}.")
    average_energy_list.append(pdc_opt.pdbo.opt_problem.simulator.cumulative_energy/96.0/len(pdc_opt.pdbo.opt_problem.evaluated_constrs_list))
    average_discomfort_list.append(pdc_opt.pdbo.opt_problem.simulator.cumulative_discomfort*0.25/len(pdc_opt.pdbo.opt_problem.evaluated_constrs_list))

discomfort_thr_to_ave_discomfort_pdcbo[discomfort_thr] = average_discomfort_list
discomfort_thr_to_ave_energy_pdcbo[discomfort_thr] = average_energy_list

discomfort_thr_to_discomfort_pdcbo[discomfort_thr] = discomfort_list
discomfort_thr_to_energy_pdcbo[discomfort_thr] = energy_list


# In[43]:


for _ in range(optimization_config['eval_budget']):
    context_vars = opt_problem.get_context(opt_problem.simulator)
    y_obj, constr_vals = pdc_opt.make_step(context_vars)
    if optimizer_type == 'safe_bo':
        new_cumu_cost = pdc_opt.safe_bo.cumu_vio_cost
    if optimizer_type == 'constrained_bo':
        new_cumu_cost = pdc_opt.constrained_bo.cumu_vio_cost
    if optimizer_type == 'violation_aware_bo':
        new_cumu_cost = pdc_opt.violation_aware_bo.cumu_vio_cost
    if optimizer_type == 'pdcbo':
        new_cumu_cost = pdc_opt.pdbo.cumu_vio_cost
    if optimizer_type == 'no opt':
        new_cumu_cost = pdc_opt.cumu_vio_cost
    if optimizer_type == 'grid search':
        new_cumu_cost = pdc_opt.cumu_vio_cost

    opt_total_cost_list.append(new_cumu_cost)
    opt_obj_list.append(y_obj)
    constraints_list.append(constr_vals)
    energy, discomfort = opt_problem.simulator.get_recent_energy_discomfort_per_day()

    energy_list.append(energy)
    discomfort_list.append(discomfort)
    print_log = True
    if print_log:
        print(f"In step {_}, with discomfort threshold " +
                f"{discomfort_thr} and discomfort weight " +
                f"{discomfort_weight}, we get energy {energy}" +
                f" and discomfort {discomfort}, with the point "
                + f" {opt_problem.evaluated_points_list[-1]}.")
    average_energy_list.append(pdc_opt.pdbo.opt_problem.simulator.cumulative_energy/96.0/len(pdc_opt.pdbo.opt_problem.evaluated_constrs_list))
    average_discomfort_list.append(pdc_opt.pdbo.opt_problem.simulator.cumulative_discomfort*0.25/len(pdc_opt.pdbo.opt_problem.evaluated_constrs_list))
print(pdc_opt.pdbo.dual)
print(np.array(average_discomfort_list))

discomfort_thr_to_ave_discomfort_pdcbo[discomfort_thr] = average_discomfort_list
discomfort_thr_to_ave_energy_pdcbo[discomfort_thr] = average_energy_list

discomfort_thr_to_discomfort_pdcbo[discomfort_thr] = discomfort_list
discomfort_thr_to_energy_pdcbo[discomfort_thr] = energy_list


# In[44]:


numerical_epsilon = 0.08
opt_problem = pdc_opt.pdbo.opt_problem
kernel = pdc_opt.pdbo.gp_obj.kern
input_x = np.array(opt_problem.evaluated_points_list).squeeze()
input_y = np.array(opt_problem.evaluated_constrs_list)[:, :, 0].squeeze().transpose()
input_cov = kernel.K(input_x) + np.eye(input_x.shape[0]) * numerical_epsilon
print(np.sqrt((np.linalg.inv(input_cov)@input_y)@input_y))
print(kernel)

for i in range(6):
    plt.figure()
    plt.scatter(input_x[:, i], input_y)


# In[45]:


average_discomfort_list = discomfort_thr_to_ave_discomfort_pdcbo[discomfort_thr]
average_energy_list = discomfort_thr_to_ave_energy_pdcbo[discomfort_thr]
plt.plot(average_discomfort_list)
plt.plot([discomfort_thr] * len(average_discomfort_list), '--')
plt.xlabel('Day')
plt.ylabel('Average discomfort per day/[K*h]')
plt.legend(['PDCBO average', 'Discomfort bound'])
plt.figure()
plt.plot(average_energy_list)
plt.xlabel('Day')
plt.ylabel('Average energy per day/[kWh]')
plt.legend(['PDCBO average'])


# In[7]:


plt.plot(pdc_opt.pdbo.opt_problem.simulator.history_dict_to_list('cumulative_discomfort', list(pdc_opt.pdbo.opt_problem.simulator.history_dict.keys())))


# In[38]:


pdc_opt.pdbo.opt_problem.evaluated_points_list


# In[8]:


pdc_opt.pdbo.opt_problem.simulator.plot_history(['tmp', 'valve_ratio', 'cumulative_energy', 'cumulative_discomfort'], 'Tmp', plot_num=4000, plot_start=1000)
plt.figure(figsize=(12, 6))
plt.plot(pdc_opt.pdbo.opt_problem.simulator.history_dict.keys(), pdc_opt.pdbo.opt_problem.simulator.history_dict_to_list('cumulative_discomfort', list(pdc_opt.pdbo.opt_problem.simulator.history_dict.keys())))

print(pdc_opt.pdbo.opt_problem.evaluated_points_list)


# # CBO

# In[9]:


cbo_config = copy.deepcopy(optimizer_base_config)
cbo_config.update({
        'total_eval_num': optimization_config['eval_budget'],
        'kernel_type': 'Gaussian'
    })

optimizer_type = 'constrained_bo'
cbo_opt, opt_total_cost_list, opt_problem = get_vacbo_optimizer(
                    cbo_config['problem_name'], optimizer_type, cbo_config,
                    discomfort_thr=discomfort_thr, vars_to_fix=VARS_TO_FIX,
                    contextual_vars=CONTEXTUAL_VARS,
                    discomfort_weight=discomfort_weight,
                    tune_obj=TUNE_OBJ, energy_thr=ENERGY_THR)
opt_obj_list = []
constraints_list = []
energy_list = []
discomfort_list = []
average_energy_list = []
average_discomfort_list = []

for _ in range(optimization_config['eval_budget']):
    context_vars = opt_problem.get_context(opt_problem.simulator)
    y_obj, constr_vals = cbo_opt.make_step(context_vars)
    if optimizer_type == 'safe_bo':
        new_cumu_cost = cbo_opt.safe_bo.cumu_vio_cost
    if optimizer_type == 'constrained_bo':
        new_cumu_cost = cbo_opt.constrained_bo.cumu_vio_cost
    if optimizer_type == 'violation_aware_bo':
        new_cumu_cost = cbo_opt.violation_aware_bo.cumu_vio_cost
    if optimizer_type == 'pdcbo':
        new_cumu_cost = cbo_opt.pdbo.cumu_vio_cost
    if optimizer_type == 'no opt':
        new_cumu_cost = cbo_opt.cumu_vio_cost
    if optimizer_type == 'grid search':
        new_cumu_cost = cbo_opt.cumu_vio_cost

    opt_total_cost_list.append(new_cumu_cost)
    opt_obj_list.append(y_obj)
    constraints_list.append(constr_vals)
    energy, discomfort = opt_problem.simulator.get_recent_energy_discomfort_per_day()

    energy_list.append(energy)
    discomfort_list.append(discomfort)
    print_log = True
    if print_log:
        print(f"In step {_}, with discomfort threshold " +
                f"{discomfort_thr} and discomfort weight " +
                f"{discomfort_weight}, we get energy {energy}" +
                f" and discomfort {discomfort}, with the point "
                + f" {opt_problem.evaluated_points_list[-1]}.")
    average_energy_list.append(cbo_opt.constrained_bo.opt_problem.simulator.cumulative_energy/96.0/len(cbo_opt.constrained_bo.opt_problem.evaluated_constrs_list))
    average_discomfort_list.append(cbo_opt.constrained_bo.opt_problem.simulator.cumulative_discomfort*0.25/len(cbo_opt.constrained_bo.opt_problem.evaluated_constrs_list))

discomfort_thr_to_ave_discomfort_cbo[discomfort_thr] = average_discomfort_list
discomfort_thr_to_ave_energy_cbo[discomfort_thr] = average_energy_list

discomfort_thr_to_discomfort_cbo[discomfort_thr] = discomfort_list
discomfort_thr_to_energy_cbo[discomfort_thr] = energy_list


# In[10]:


average_discomfort_list = discomfort_thr_to_ave_discomfort_cbo[discomfort_thr]
average_energy_list = discomfort_thr_to_ave_energy_cbo[discomfort_thr]
plt.plot(average_discomfort_list)
plt.plot([discomfort_thr] * len(average_discomfort_list), '--')
plt.xlabel('Day')
plt.ylabel('Average discomfort per day/[K*h]')
plt.legend(['CBO average', 'Discomfort bound'])
plt.figure()
plt.plot(average_energy_list)
plt.xlabel('Day')
plt.ylabel('Average energy per day/[kWh]')
plt.legend(['CBO average'])


# # Safe BO

# In[11]:


safebo_config = copy.deepcopy(optimizer_base_config)
safebo_config.update({
        'total_eval_num': optimization_config['eval_budget'],
        'kernel_type': 'Gaussian'
    })

optimizer_type = 'safe_bo'
safe_opt, opt_total_cost_list, opt_problem = get_vacbo_optimizer(
                    cbo_config['problem_name'], optimizer_type, safebo_config,
                    discomfort_thr=discomfort_thr, vars_to_fix=VARS_TO_FIX,
                    contextual_vars=CONTEXTUAL_VARS,
                    discomfort_weight=discomfort_weight,
                    tune_obj=TUNE_OBJ, energy_thr=ENERGY_THR)
opt_obj_list = []
constraints_list = []
energy_list = []
discomfort_list = []
average_energy_list = []
average_discomfort_list = []

for _ in range(optimization_config['eval_budget']):
    context_vars = opt_problem.get_context(opt_problem.simulator)
    y_obj, constr_vals = safe_opt.make_step(context_vars)
    if optimizer_type == 'safe_bo':
        new_cumu_cost = safe_opt.safe_bo.cumu_vio_cost
    if optimizer_type == 'constrained_bo':
        new_cumu_cost = safe_opt.constrained_bo.cumu_vio_cost
    if optimizer_type == 'violation_aware_bo':
        new_cumu_cost = safe_opt.violation_aware_bo.cumu_vio_cost
    if optimizer_type == 'pdcbo':
        new_cumu_cost = safe_opt.pdbo.cumu_vio_cost
    if optimizer_type == 'no opt':
        new_cumu_cost = safe_opt.cumu_vio_cost
    if optimizer_type == 'grid search':
        new_cumu_cost = safe_opt.cumu_vio_cost

    opt_total_cost_list.append(new_cumu_cost)
    opt_obj_list.append(y_obj)
    constraints_list.append(constr_vals)
    energy, discomfort = opt_problem.simulator.get_recent_energy_discomfort_per_day()

    energy_list.append(energy)
    discomfort_list.append(discomfort)
    print_log = True
    if print_log:
        print(f"In step {_}, with discomfort threshold " +
                f"{discomfort_thr} and discomfort weight " +
                f"{discomfort_weight}, we get energy {energy}" +
                f" and discomfort {discomfort}, with the point "
                + f" {opt_problem.evaluated_points_list[-1]}.")
    average_energy_list.append(safe_opt.safe_bo.opt_problem.simulator.cumulative_energy/96.0/len(safe_opt.safe_bo.opt_problem.evaluated_constrs_list))
    average_discomfort_list.append(safe_opt.safe_bo.opt_problem.simulator.cumulative_discomfort*0.25/len(safe_opt.safe_bo.opt_problem.evaluated_constrs_list))

discomfort_thr_to_ave_discomfort_safe_bo[discomfort_thr] = average_discomfort_list
discomfort_thr_to_ave_energy_safe_bo[discomfort_thr] = average_energy_list

discomfort_thr_to_discomfort_safe_bo[discomfort_thr] = discomfort_list
discomfort_thr_to_energy_safe_bo[discomfort_thr] = energy_list


# In[12]:


average_discomfort_list = discomfort_thr_to_ave_discomfort_safe_bo[discomfort_thr]
average_energy_list = discomfort_thr_to_ave_energy_safe_bo[discomfort_thr]
plt.plot(average_discomfort_list)
plt.plot([discomfort_thr] * len(average_discomfort_list), '--')
plt.xlabel('Day')
plt.ylabel('Average discomfort per day/[K*h]')
plt.legend(['Safe BO average', 'Discomfort bound'])
plt.figure()
plt.plot(average_energy_list)
plt.xlabel('Day')
plt.ylabel('Average energy per day/[kWh]')
plt.legend(['Safe BO average'])


# # fixed parameter

# In[46]:


no_opt_config = copy.deepcopy(optimizer_base_config)
no_opt_config.update({
        'total_eval_num': optimization_config['eval_budget']
    })

optimizer_type = 'no opt'
fix_opt, opt_total_cost_list, opt_problem = get_vacbo_optimizer(
                    no_opt_config['problem_name'], optimizer_type, no_opt_config,
                    discomfort_thr=discomfort_thr, vars_to_fix=VARS_TO_FIX,
                    contextual_vars=CONTEXTUAL_VARS,
                    discomfort_weight=discomfort_weight,
                    tune_obj=TUNE_OBJ, energy_thr=ENERGY_THR)
opt_obj_list = []
constraints_list = []
energy_list = []
discomfort_list = []
average_energy_list = []
average_discomfort_list = []

for _ in range(optimization_config['eval_budget']):
    context_vars = opt_problem.get_context(opt_problem.simulator)
    y_obj, constr_vals = fix_opt.make_step()
    if optimizer_type == 'safe_bo':
        new_cumu_cost = fix_opt.safe_bo.cumu_vio_cost
    if optimizer_type == 'constrained_bo':
        new_cumu_cost = fix_opt.constrained_bo.cumu_vio_cost
    if optimizer_type == 'violation_aware_bo':
        new_cumu_cost = fix_opt.violation_aware_bo.cumu_vio_cost
    if optimizer_type == 'pdcbo':
        new_cumu_cost = fix_opt.pdbo.cumu_vio_cost
    if optimizer_type == 'no opt':
        new_cumu_cost = fix_opt.cumu_vio_cost
    if optimizer_type == 'grid search':
        new_cumu_cost = fix_opt.cumu_vio_cost

    opt_total_cost_list.append(new_cumu_cost)
    opt_obj_list.append(y_obj)
    constraints_list.append(constr_vals)
    energy, discomfort = opt_problem.simulator.get_recent_energy_discomfort_per_day()

    energy_list.append(energy)
    discomfort_list.append(discomfort)
    print_log = True
    if print_log:
        print(f"In step {_}, with discomfort threshold " +
                f"{discomfort_thr} and discomfort weight " +
                f"{discomfort_weight}, we get energy {energy}" +
                f" and discomfort {discomfort}, with the point "
                + f" {opt_problem.evaluated_points_list[-1]}.")
    average_energy_list.append(fix_opt.opt_problem.simulator.cumulative_energy/96.0/len(fix_opt.opt_problem.evaluated_constrs_list))
    average_discomfort_list.append(fix_opt.opt_problem.simulator.cumulative_discomfort*0.25/len(fix_opt.opt_problem.evaluated_constrs_list))

discomfort_thr_to_ave_discomfort_no_opt[discomfort_thr] = average_discomfort_list
discomfort_thr_to_ave_energy_no_opt[discomfort_thr] = average_energy_list

discomfort_thr_to_discomfort_no_opt[discomfort_thr] = discomfort_list
discomfort_thr_to_energy_no_opt[discomfort_thr] = energy_list


# In[14]:


average_discomfort_list = discomfort_thr_to_ave_discomfort_no_opt[discomfort_thr]
average_energy_list = discomfort_thr_to_ave_energy_no_opt[discomfort_thr]
plt.plot(average_discomfort_list)
plt.plot([discomfort_thr] * len(average_discomfort_list), '--')
plt.xlabel('Day')
plt.ylabel('Average discomfort per day/[K*h]')
plt.legend(['No opt average', 'Discomfort bound'])
plt.figure()
plt.plot(np.array(average_energy_list) * 1000)
plt.xlabel('Day')
plt.ylabel('Average energy per day/[kWh]')
plt.legend(['No opt average'])


# # comparison

# In[15]:


average_discomfort_list_pd = discomfort_thr_to_ave_discomfort_pdcbo[discomfort_thr]
average_energy_list_pd = discomfort_thr_to_ave_energy_pdcbo[discomfort_thr]

average_discomfort_list_cbo = discomfort_thr_to_ave_discomfort_cbo[discomfort_thr]
average_energy_list_cbo = discomfort_thr_to_ave_energy_cbo[discomfort_thr]

average_discomfort_list_safe = discomfort_thr_to_ave_discomfort_safe_bo[discomfort_thr]
average_energy_list_safe = discomfort_thr_to_ave_energy_safe_bo[discomfort_thr]

plt.plot(average_discomfort_list_cbo)
plt.plot(average_discomfort_list_safe)
plt.plot(average_discomfort_list_pd)
plt.plot([discomfort_thr] * len(average_discomfort_list), '--')
plt.xlabel('Day')
plt.ylabel('Average discomfort per day/[K*h]')
plt.legend(['CBO', 'Safe BO', 'PDCBO', 'Discomfort bound'])
plt.savefig(f'./fig/average_discomfort_with_pdcbo_{discomfort_thr}.pdf', format='pdf')

plt.figure()

plt.plot(np.array(average_energy_list_cbo)*1000)
plt.plot(np.array(average_energy_list_safe)*1000)
plt.plot(np.array(average_energy_list_pd)*1000)
plt.xlabel('Day')
plt.ylabel('Average energy per day/[kWh]')
plt.legend(['cBO', 'Safe BO', 'PDCBO'])
plt.savefig(f'./fig/average_energy_with_pdcbo_{discomfort_thr}.pdf', format='pdf')


# In[16]:


start = 6000
data_num = 1000
pdc_opt.pdbo.opt_problem.simulator.plot_history(['tmp', 'valve_ratio'], title_str='PDCBO', plot_num=start+data_num, plot_start=start)
#safe_opt.safe_bo.opt_problem.simulator.plot_history(['tmp'], title_str='SafeBO', plot_num=3000, plot_start=0)
#cbo_opt.constrained_bo.opt_problem.simulator.plot_history(['tmp'], title_str='CBO')

fix_opt.opt_problem.simulator.plot_history(['tmp', 'valve_ratio'], title_str='Fixed Solution', plot_num=start+data_num, plot_start=start)


# In[ ]:





# In[ ]:





# In[ ]:




