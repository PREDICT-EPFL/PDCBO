"""
Define the temperature cost function.
"""


def temp_discomfort_cost(x, time_int=1):
    comfort_lb = 23
    comfort_ub = 24

    comfort_relaxed_lb = 21
    comfort_relaxed_ub = 24

    start_work_hour = 7
    end_work_hour = 19

    date_time = x[0]
    temp = x[1]
    hour = date_time.hour
    cost = 0

    if start_work_hour <= hour <= end_work_hour:
        comfort_range = [comfort_lb, comfort_ub]
        if temp > comfort_ub:
            cost = (temp - comfort_ub) * time_int
            # in the unit of K * h
        elif temp < comfort_lb:
            cost = (comfort_lb - temp) * time_int
            # in the unit of K * h
    else:
        comfort_range = [comfort_relaxed_lb, comfort_relaxed_ub]
        if temp > comfort_relaxed_ub:
            cost = (temp - comfort_relaxed_ub) * time_int
            # in the unit of K * h
        elif temp < comfort_relaxed_lb:
            cost = (comfort_relaxed_lb - temp) * time_int
            # in the unit of K * h
    return cost, comfort_range


def get_dict_medium(tuple_dict):
    medium_dict = dict()
    for key in tuple_dict.keys():
        two_tuple = tuple_dict[key]
        medium_dict[key] = (two_tuple[0] + two_tuple[1]) * 0.5

    return medium_dict
