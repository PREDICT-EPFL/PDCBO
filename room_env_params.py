import datetime as dt

def get_room_env_params():
    LinearRoomParams = {
            'name': '272',
            'T_room': 0.9827885835486007,
            'uk': 0.08968639447111469,
            'T_neighbors': [0.014980144404468954],
            'T_amb': 0.002715099486652448,
            's_coefs': [1.6551779687278937e-06, 9.28778618279388e-06,
                        9.596021706005906e-05, 0.00010865884345802481,
                        6.675261810419619e-07, -3.4765761711581095e-05,
                        -9.301848292875373e-06, 1.353656562614565e-06,
                        9.603471638607937e-07]
    }
    STEP_DURATION = 15 * 60  # unit: second
    MAX_VALVE_POWER = 1.0  # depending on how much power
    start_date_time = dt.datetime.strptime('2018-12-01 00:00:00',
                                           '%Y-%m-%d %H:%M:%S')
    ambient_file = \
        './EMPA_VABO_experiments/ambient_data/ambientconditions_15.csv'
    return LinearRoomParams, STEP_DURATION, MAX_VALVE_POWER, start_date_time, \
        ambient_file
