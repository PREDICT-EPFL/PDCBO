"""
Implement violation-aware Bayesian optimizer.
"""
import numpy as np
import safeopt
import copy
import vacbo
import util
from keep_default_optimizer import KeepDefaultOpt


DISCOMFORT_THR = 10
optimization_config = {
    'eval_budget': 40
}

optimizer_base_config = {
    'noise_level': [0.004, 0.2, 0.2],
    'kernel_var': 0.1,
    'train_noise_level': 1.0,
    'problem_name': 'SinglePIRoomEvaluator',
    'normalize_input': False
}

VARS_TO_FIX = ['high_on_time', 'high_off_time', 'high_setpoint',
               'low_setpoint', 'control_setpoint']
CONTEXTUAL_VARS = ['Q_irr', 'T_out', 'T_init']


def get_no_opt_optimizer(problem_name, optimizer_type, optimizer_config,
                         init_points_id=0, discomfort_thr=2.0,
                         vars_to_fix=None, contextual_vars=None,
                         start_date_time=None, fixed_param=None,
                         discomfort_weight=0.01, tune_var_scale='log'):
    problem_config = util.get_config(
        problem_name, gp_kernel='Matern52', init_points_id=init_points_id,
        discomfort_thr=discomfort_thr, vars_to_fix=VARS_TO_FIX,
        start_eval_time=start_date_time, room_simulator='PCNN',
        discomfort_weight=discomfort_weight, tune_PI_scale=tune_var_scale,
        contextual_vars=CONTEXTUAL_VARS)
    optimizer_config.update(problem_config)
    if fixed_param is not None:
        problem_config['init_safe_points'] = fixed_param
    problem = vacbo.ContextualOptimizationProblem(problem_config)
    if optimizer_type == 'no opt':
        opt = KeepDefaultOpt(problem, optimizer_config)
        total_cost_list = [opt.cumu_vio_cost]

    return opt, total_cost_list, problem


def evaluate_seq_control(fixed_param, discomfort_thr,
                         control_seq, discomfort_weight=0.01,
                         optimizer_base_config=None,
                         optimization_config=None,
                         vars_to_fix=None):
    # try fixing one parameter
    no_opt_config = copy.deepcopy(optimizer_base_config)

    no_opt, no_opt_best_obj_list, no_opt_total_cost_list = \
        get_no_opt_optimizer(
            no_opt_config['problem_name'], 'no opt', no_opt_config,
            discomfort_thr=discomfort_thr, vars_to_fix=vars_to_fix,
            fixed_param=fixed_param, discomfort_weight=discomfort_weight)
    num_of_control = len(control_seq[:, 0])
    for _ in range(num_of_control):
        y_obj, constr_vals = no_opt.make_step(
            np.expand_dims(control_seq[_, :], axis=0))
    simulator = no_opt.opt_problem.simulator
    simulator.update_history_dict()
    cumulative_discomfort = simulator.cumulative_discomfort
    date_time_list = list(simulator.history_dict.keys())
    cumulative_energy = sum(
        simulator.history_dict_to_list('power room', date_time_list)
    ) * 0.001 * 0.25
    num_of_data = len(date_time_list)
    energy_per_day = cumulative_energy / (num_of_data / 96)
    ave_discomfort = cumulative_discomfort / num_of_data
    return energy_per_day, ave_discomfort, cumulative_energy, \
        cumulative_discomfort


class SeqGridSearchOpt:

    def __init__(self, opt_problem, keep_default_config):

        self.current_step = 0
        self.opt_problem = opt_problem
        self.noise_level = keep_default_config['noise_level']
        self.current_step = 0
        self.cumu_vio_cost = 0
        # Bounds on the inputs variable
        self.bounds = opt_problem.bounds
        self.discret_num_list = opt_problem.discretize_num_list

        # set of parameters
        self.parameter_set = safeopt.linearly_spaced_combinations(
            self.bounds,
            self.discret_num_list
        )

        # Initial safe point
        self.x0_arr = opt_problem.init_safe_points
        self.query_points_list = []
        self.query_point_obj = []
        self.query_point_constrs = []
        self.S = []
        # self.kernel_list = []
        init_obj_val_arr, init_constr_val_arr = \
            self.get_obj_constr_val(self.x0_arr)
        self.init_obj_val_arr = init_obj_val_arr
        self.init_constr_val_arr = init_constr_val_arr
        self.best_obj = np.min(init_obj_val_arr)
        best_obj_id = np.argmin(init_obj_val_arr[:, 0])
        self.best_sol = self.x0_arr[best_obj_id, :]
        self.evaluate_id = 0
        if 'discomfort_weight' in keep_default_config.keys():
            self.discomfort_weight = keep_default_config['discomfort_weight']

    def get_obj_constr_val(self, x_arr, noise=False):
        obj_val_arr, constr_val_arr = self.opt_problem.sample_point(x_arr)
        return obj_val_arr, constr_val_arr

    def plot(self):
        # Plot the GP
        self.opt.plot(100)
        # Plot the true function
        y, constr_val = self.get_obj_constr_val(self.parameter_set,
                                                noise=False)

    def look_ahead_step(self):
        current_control_seq = np.atleast_2d(
            np.squeeze(np.array(self.query_points_list))
        )
        total_controller_num = len(self.parameter_set[:, 0])
        aug_energy_list = []
        aug_discomfort_list = []
        for controller_id in range(total_controller_num):
            new_control = np.expand_dims(
                self.parameter_set[controller_id, :], axis=0)
            # print(new_control, current_control_seq)
            energy_consumption, discomfort, max_discomfort = \
                self.opt_problem.obj(
                    new_control,
                    simulator_to_use=self.opt_problem.simulator,
                    is_look_ahead=True
                )
            total_discomfort = \
                self.opt_problem.simulator.cumulative_discomfort + discomfort
            total_energy = \
                self.opt_problem.simulator.cumulative_energy + \
                energy_consumption
            print(new_control, total_energy, total_discomfort)
            aug_energy_list.append(total_energy)
            aug_discomfort_list.append(total_discomfort)

        # optimization criterion
        discomfort_weight = self.discomfort_weight
        aug_energy_arr = np.array(aug_energy_list)
        aug_discomfort_arr = np.array(aug_discomfort_list)
        weighted_min_obj = aug_energy_arr + \
            discomfort_weight * aug_discomfort_arr
        opt_id = np.argmin(weighted_min_obj)
        opt_controller = np.array(np.expand_dims(
            self.parameter_set[opt_id, :], axis=0))
        print(aug_energy_arr,
              aug_discomfort_arr,
              discomfort_weight,
              weighted_min_obj)

        return opt_controller

    def make_step(self, evaluate_point=None):
        self.current_step += 1
        if evaluate_point is None:
            x_next = self.look_ahead_step()
        else:
            x_next = evaluate_point
        # Get a measurement from the real system
        # print(evaluate_point)
        # print(x_next)
        y_obj, constr_vals = self.get_obj_constr_val(x_next)

        self.query_points_list.append(x_next)
        self.query_point_obj.append(y_obj)
        self.query_point_constrs.append(constr_vals)

        vio_cost = self.opt_problem.get_total_violation_cost(constr_vals)
        vio_cost = np.squeeze(vio_cost)
        if np.all(constr_vals <= 0):
            # update best objective if we get a feasible point
            if self.best_obj > y_obj[0, 0]:
                self.best_sol = x_next
            self.best_obj = np.min([y_obj[0, 0], self.best_obj])
        violation_cost = self.opt_problem.get_total_violation_cost(constr_vals)
        violation_total_cost = np.sum(violation_cost, axis=0)
        self.cumu_vio_cost = self.cumu_vio_cost + violation_total_cost

        return y_obj, constr_vals
