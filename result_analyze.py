import numpy as np


# obj_list_dict, constrain_list_dict, energy_list_dict, \
#       discomfort_list_dict, seasonal_energy_list_dict, \
#        seasonal_discomfort_list_dict

def read_result(data_path):
    return list(x.item() for x in
                np.load(data_path, allow_pickle=True).values())

def turn_array_list_to_compact_array(array_list):
    return np.array(array_list).squeeze()

def dict_to_same_key_values(dict_iter, key):
    print(dict_iter[0].keys())
    return [iter_dict[key] for iter_dict in dict_iter]

def read_one_key_result(data_path, discomfort_thr):
    return list(turn_array_list_to_compact_array(x) for
                x in dict_to_same_key_values(
                    read_result(data_path), discomfort_thr)
                )

def get_result_keys(data_path):
    return list(read_result(data_path)[0].keys())
