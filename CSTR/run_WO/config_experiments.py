import time
import random
import numpy as np
import numpy.random as rnd
from scipy.spatial.distance import cdist

import sobol_seq
from scipy.optimize import minimize
from scipy.optimize import broyden1
from scipy import linalg
import scipy
import matplotlib.pyplot as plt
import functools
from matplotlib.patches import Ellipse

from casadi import *

from sub_uts.systems import *
from sub_uts.utilities_2 import *
from plots_RTO import compute_obj, plot_obj, plot_obj_noise

import pickle
from plots_RTO import Plot
#----------1) EI-NO PRIOR-UNKNOWN NOISE----------#
#---------------------------------------------#
if not(os.path.exists('figs_WO')):
    os.mkdir('figs_WO')
if not(os.path.exists('figs_noise_WO')):
    os.mkdir('figs_noise_WO')

# obj_no_prior_with_exploration_ei          = compute_obj('no_prior_with_exploration_ei')
# obj_with_prior_with_exploration_ei        = compute_obj('with_prior_with_exploration_ei')
# obj_with_prior_with_exploration_ei_noise  = compute_obj('with_prior_with_exploration_ei_noise')
# obj_no_prior_with_exploration_ei_noise    = compute_obj('no_prior_with_exploration_ei_noise')
# obj_no_prior_with_exploration_ucb         = compute_obj('no_prior_with_exploration_ucb')
# obj_with_prior_with_exploration_ucb       = compute_obj('with_prior_with_exploration_ucb')
# obj_with_prior_with_exploration_ucb_noise = compute_obj('with_prior_with_exploration_ucb_noise')
# obj_no_prior_with_exploration_ucb_noise   = compute_obj('no_prior_with_exploration_ucb_noise')
# obj_no_prior_no_exploration               = compute_obj('no_prior_no_exploration')
# obj_with_prior_no_exploration             = compute_obj('with_prior_no_exploration')
# obj_with_prior_no_exploration_noise       = compute_obj('with_prior_no_exploration_noise')
# obj_no_prior_no_exploration_noise         = compute_obj('no_prior_no_exploration_noise')
#
# data = [obj_no_prior_with_exploration_ei[-1],
#         obj_with_prior_with_exploration_ei[-1],
#         obj_with_prior_with_exploration_ei_noise[-1],
#         obj_no_prior_with_exploration_ei_noise[-1],
#         obj_no_prior_with_exploration_ucb[-1],
#         obj_with_prior_with_exploration_ucb[-1],
#         obj_with_prior_with_exploration_ucb_noise[-1],
#         obj_no_prior_with_exploration_ucb_noise[-1],
#         obj_no_prior_no_exploration[-1],
#         obj_with_prior_no_exploration[-1],
#         obj_with_prior_no_exploration_noise[-1],
#         obj_no_prior_no_exploration_noise[-1]]
# ni = 20
# for i,obj_ in enumerate(data):
#     obj_mean = obj_.mean(axis=0)
#     obj_max = obj_.max(axis=0)
#     obj_min = obj_.min(axis=0)
#     plt.errorbar(np.linspace(1, ni, ni), obj_mean, yerr=[obj_mean - obj_min, obj_max - obj_mean],
#              alpha=1.)
# #
# # plt.plot(np.linspace(1, ni, ni), obj_max,
# #              color='#255E69', alpha=1.)
# #
# # plt.plot(np.linspace(1, ni, ni), obj_min,
# #              color='#255E69', alpha=1.)
# # plt.plot(np.linspace(1, ni, ni), [obj_max.max()]*ni,
# #              color='#255E69', alpha=1.)
# plt.xlabel('RTO-iter')
# plt.ylabel('Objective')
# plt.tick_params(right=True, top=True, left=True, bottom=True)
# plt.tick_params(axis="y", direction="in")
# plt.tick_params(axis="x", direction="in")
# plt.tight_layout()
# plt.savefig('obj.png', dpi=400)
# plt.close()

# Plot('no_prior_with_exploration_ei')
# Plot('with_prior_with_exploration_ei')
# Plot('with_prior_with_exploration_ei_noise')
# Plot('no_prior_with_exploration_ei_noise')
# Plot('no_prior_with_exploration_ucb')
# Plot('with_prior_with_exploration_ucb')
# Plot('with_prior_with_exploration_ucb_noise')
# Plot('no_prior_with_exploration_ucb_noise')
# Plot('no_prior_no_exploration')
# Plot('with_prior_no_exploration')
# Plot('with_prior_no_exploration_noise')
# Plot('no_prior_no_exploration_noise')
plot_obj(compute_obj)
# plot_obj_noise(compute_obj)

#-----------------------------------------------------------------------#
#----------2) EI-PRIOR-UNKNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):

    model = WO_model()
    plant = WO_system()

    obj_model      = model.WO_obj_ca
    cons_model     = [model.WO_con1_model_ca, model.WO_con2_model_ca]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = None#[0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=1, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('with_prior_no_exploration_new2.p','wb'))











np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):

    plant = WO_system()

    obj_model      = obj_empty#model.WO_obj_ca
    cons_model     = [con_empty, con_empty]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = None#[0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=3, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('no_prior_with_exploration_ei.p','wb'))
#-----------------------------------------------------------------------#
#----------3) EI-PRIOR-KNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):

    model = WO_model()
    plant = WO_system()

    obj_model      = model.WO_obj_ca
    cons_model     = [model.WO_con1_model_ca, model.WO_con2_model_ca]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=3, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('with_prior_with_exploration_ei_noise.p','wb'))
#-----------------------------------------------------------------------#
#----------4) EI-NO PRIOR-KNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):


    plant = WO_system()

    obj_model      = obj_empty
    cons_model     = [con_empty, con_empty]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=3, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('no_prior_with_exploration_ei_noise.p','wb'))
#-----------------------------------------------------------------------#


#-----------------------------UCB----------------------------------------------#
#----------1) UCB-NO PRIOR-UNKNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):

    plant = WO_system()

    obj_model      = obj_empty#model.WO_obj_ca
    cons_model     = [con_empty, con_empty]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = None#[0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=2, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('no_prior_with_exploration_ucb.p','wb'))
#-----------------------------------------------------------------------#
#----------2) UCB-PRIOR-UNKNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):

    model = WO_model()
    plant = WO_system()

    obj_model      = model.WO_obj_ca
    cons_model     = [model.WO_con1_model_ca, model.WO_con2_model_ca]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = None#[0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=2, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('with_prior_with_exploration_ucb.p','wb'))
#-----------------------------------------------------------------------#
#----------3) UCB-PRIOR-KNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):

    model = WO_model()
    plant = WO_system()

    obj_model      = model.WO_obj_ca
    cons_model     = [model.WO_con1_model_ca, model.WO_con2_model_ca]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=2, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('with_prior_with_exploration_ucb_noise.p','wb'))
#-----------------------------------------------------------------------#
#----------4) UCB-NO PRIOR-KNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):


    plant = WO_system()

    obj_model      = obj_empty
    cons_model     = [con_empty, con_empty]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=2, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('no_prior_with_exploration_ucb_noise.p','wb'))
#-----------------------------------------------------------------------#


#-----------------------------No exploration----------------------------------------------#
#----------1) noexplore-NO PRIOR-UNKNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):

    plant = WO_system()

    obj_model      = obj_empty#model.WO_obj_ca
    cons_model     = [con_empty, con_empty]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = None#[0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=1, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('no_prior_no_exploration.p','wb'))
#-----------------------------------------------------------------------#
#----------2) noexplorePRIOR-UNKNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):

    model = WO_model()
    plant = WO_system()

    obj_model      = model.WO_obj_ca
    cons_model     = [model.WO_con1_model_ca, model.WO_con2_model_ca]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = None#[0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=1, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('with_prior_no_exploration.p','wb'))
#-----------------------------------------------------------------------#
#----------3) no exploration-PRIOR-KNOWN NOISE----------#
#---------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):

    model = WO_model()
    plant = WO_system()

    obj_model      = model.WO_obj_ca
    cons_model     = [model.WO_con1_model_ca, model.WO_con2_model_ca]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=1, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('with_prior_no_exploration_noise.p','wb'))
#-----------------------------------------------------------------------#
#----------4) Noexplore-NO PRIOR-KNOWN NOISE----------#
#---------------------------------------------#
#----------------------------------------------#
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(30):


    plant = WO_system()

    obj_model      = obj_empty
    cons_model     = [con_empty, con_empty]
    obj_system     = plant.WO_obj_sys_ca
    cons_system    = [plant.WO_con1_sys_ca, plant.WO_con2_sys_ca]




    n_iter         = 20
    bounds         = [[4.,7.],[70.,100.]]
    Xtrain         = np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = np.array([6.9,83])

    Delta0         = 0.25
    Delta_max      =0.7; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2;
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = [0.5**2, 5e-8, 5e-8]


    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=1, noise=noise, multi_opt=30,
                                    multi_hyper=15, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=True)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]








