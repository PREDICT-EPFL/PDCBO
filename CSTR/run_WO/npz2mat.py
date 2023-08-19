import scipy
import numpy as np
import os

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

def npz2mat(npz_file):
    total_cost_list, best_obj_list, opt_X, opt_points, opt_contrs, opt_objs = read_simulation_data(npz_file)
    mdict = dict()
    mdict['total_cost_list'] = total_cost_list
    mdict['best_obj_list'] = best_obj_list
    mdict['opt_X'] = opt_X
    mdict['opt_points'] = opt_points
    mdict['opt_contrs'] = opt_contrs
    mdict['opt_objs'] = opt_objs
    scipy.io.savemat(npz_file[:-4]+'.mat', mdict)

def allnpz2mat(dir_path):
    npz_file_list = [x for x in os.listdir(dir_path) if x.endswith('.npz')]
    for npz_file_name in npz_file_list:
        npz_file = dir_path+'/'+npz_file_name
        npz2mat(npz_file)




