def get_rb_room_controller_vars_len_scales():
    """
    5 vars to tune:
    open up temperature
    shut down temperature
    input normalized power
    start working time
    end working time
    """
    all_vars = ['open_temp', 'close_temp', 'input_power', 'start_time',
                'end_time']
    all_vars_bounds_dict = {'open_temp': (17, 22),
                            'close_temp': (22, 24),
                            'input_power': (0.4, 1.0),
                            'start_time': [3 * 60, 7 * 60],
                            'end_time': [17 * 60, 19 * 60]
                            }
    all_vars_safe_dict = {'open_temp': 21,
                          'close_temp': 22.5,
                          'input_power': 1.0,
                          'start_time': 3 * 60,
                          'end_time': 19 * 60
                          }
    all_vars_energy_func_len_scales = {'open_temp': 4.0,
                                       'close_temp': 4.0,
                                       'input_power': 1.0,
                                       'start_time': 20,
                                       'end_time': 100
                                       }
    all_vars_discomfort_func_len_scales = {'open_temp': 5.0,
                                           'close_temp': 2.0,
                                           'input_power': 0.3,
                                           'start_time': 60,
                                           'end_time': 50
                                           }
    all_vars_max_discomfort_func_len_scales = {'open_temp': 20.0,
                                               'close_temp': 4.0,
                                               'input_power': 1.0,
                                               'start_time': 35,
                                               'end_time': 50
                                               }
    all_vars_default_vals = get_dict_medium(all_vars_bounds_dict)

    # set the default close_tmp
    all_vars_default_vals['close_temp'] = 22.5
    all_vars_default_vals['end_time'] = 19 * 60
    return all_vars, all_vars_bounds_dict, all_vars_safe_dict, \
        all_vars_energy_func_len_scales, all_vars_discomfort_func_len_scales, \
        all_vars_max_discomfort_func_len_scales, all_vars_default_vals


