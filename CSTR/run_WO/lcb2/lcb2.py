"""
Implement violation-aware Bayesian optimizer.
"""
import numpy as np
from .base_optimizer import BaseBO
from scipy.stats import norm


class ViolationAwareBO(BaseBO):

    def __init__(self, opt_problem, violation_aware_BO_config):
        # optimization problem and measurement noise
        super().__init__(opt_problem, violation_aware_BO_config)

        # Pr(cost <= beta * budget) >= 1 - \epsilon
        if 'beta_func' in violation_aware_BO_config.keys():
            self.beta_func = violation_aware_BO_config['beta_func']
        else:
            self.beta_func = lambda t: 1

        if 'lcb_coef' in violation_aware_BO_config.keys():
            self.lcb_coef = violation_aware_BO_config['lcb_coef']
        else:
            self.lcb_coef = lambda t: 3

        self.INF = 1e10
        self.num_eps = 1e-10   # epsilon for numerical value
        self.total_vio_budgets = violation_aware_BO_config['total_vio_budgets']
        self.increase_budget = self.total_vio_budgets / 8.0
        self.prob_eps = violation_aware_BO_config['prob_eps']
        self.beta_0 = violation_aware_BO_config['beta_0']
        self.total_eval_num = violation_aware_BO_config['total_eval_num']

        self.curr_budgets = self.total_vio_budgets
        self.curr_eval_budget = self.total_eval_num
        self.single_step_budget = violation_aware_BO_config[
               'single_max_budget']
        if 'acq_func_type' not in violation_aware_BO_config.keys():
            self.acq_func_type = 'CEI'  # default acquistion function type
        else:
            self.acq_func_type = violation_aware_BO_config['acq_func_type']
        self.cumu_vio_cost = np.zeros(self.opt_problem.num_constrs)

        if 'break_up_vio_cost' in violation_aware_BO_config.keys():
            self.break_vio_cost = violation_aware_BO_config['break_vio_cost']
        else:
            self.break_vio_cost = 100 * violation_aware_BO_config[
                'total_vio_budgets']

        if 'back_up_vio_cost' in violation_aware_BO_config.keys():
            self.back_up_vio_cost = violation_aware_BO_config[
                'back_up_vio_cost']
        else:
            self.back_up_vio_cost = 10 * violation_aware_BO_config[
                'total_vio_budgets']

        self.S = None

    def get_acquisition(self, prob_eps=None, acq_func_type='CEI'):
        if prob_eps is None:
            prob_eps = self.prob_eps
        obj_mean, obj_var = self.gp_obj.predict(self.parameter_set)
        obj_mean = obj_mean + self.gp_obj_mean
        obj_mean = obj_mean.squeeze()
        obj_var = obj_var.squeeze()
        constrain_mean_list = []
        constrain_var_list = []
        for i in range(self.opt_problem.num_constrs):
            mean, var = self.gp_constr_list[i].predict(self.parameter_set)
            mean = mean + self.gp_constr_mean_list[i]
            constrain_mean_list.append(np.squeeze(mean))
            constrain_var_list.append(np.squeeze(var))

        constrain_mean_arr = np.array(constrain_mean_list).T
        constrain_var_arr = np.array(constrain_var_list).T

        # calculate Pr(g_i(x)<=0)
        prob_negative = norm.cdf(0, constrain_mean_arr, constrain_var_arr)
        # calculate feasibility prob
        prob_feasible = np.prod(prob_negative, axis=1)

        # calculate EI and EIc
        if acq_func_type == 'CpEI':
            # for CpEI, use the posterior mean as the proxy minimum obj
            # observed
            f_min = np.min(obj_mean)
        else:
            f_min = self.best_obj

        z = (f_min - obj_mean)/np.maximum(np.sqrt(obj_var), self.num_eps)
        EI = (f_min - obj_mean) * norm.cdf(z) + np.sqrt(obj_var) * norm.pdf(z)
        EIc = prob_feasible * EI

        # calculate LCB
        trunc_obj_sd = np.maximum(np.sqrt(obj_var), self.num_eps)
        lcb_coef = self.lcb_coef(
            self.total_eval_num - self.curr_eval_budget)
        lcb = obj_mean - lcb_coef * trunc_obj_sd

        # calculate Pr(c_i([g_i(x)]^+)<=B_{i,t} * beta_{i, t})
        curr_beta = self.get_beta()
        curr_cost_allocated = self.curr_budgets * curr_beta
        allowed_vio = self.opt_problem.get_vio_from_cost(curr_cost_allocated)
        prob_not_use_up_budget = norm.cdf(allowed_vio, constrain_mean_arr,
                                          constrain_var_arr)
        prob_all_not_use_up_budget = np.prod(prob_not_use_up_budget, axis=1)

        feasible = (prob_all_not_use_up_budget >= 1 - prob_eps)
        EIc_indicated = EIc * feasible
        self.S = self.parameter_set[(prob_all_not_use_up_budget >=
                                     1 - prob_eps)]

        # for LCB, we are minimizing, set objetive to be lcb * 1_feasible +
        # INF * (1 - 1_feasible)
        lcb_indicated = lcb * feasible + self.INF * (1-feasible)

        if acq_func_type == 'CEI' or acq_func_type == 'CpEI':
            obj_indicated = EIc_indicated
        elif acq_func_type == 'LCB':
            obj_indicated = lcb_indicated
        return obj_indicated

    def get_beta(self):
        return min(max(0, self.beta_func(self.curr_eval_budget)), 1.0)

    def optimize(self, expand_safe_set_method='budget'):
        acq_func_type = self.acq_func_type
        prob_eps = self.prob_eps
        eps_multi = 1.1
        is_any_acq_postive = False
        if acq_func_type == 'CEI' or acq_func_type == 'CpEI':
            while not is_any_acq_postive:
                acq = self.get_acquisition(prob_eps=prob_eps)
                if np.any(acq > 0):
                    is_any_acq_postive = True
                else:
                    if expand_safe_set_method == 'risk':
                        print('Can not find not use up budget point, ' +
                              'increase risk level.')
                        prob_eps = prob_eps * eps_multi
                    elif expand_safe_set_method == 'budget':
                        print('Can not find not use up budget point, ' +
                              'increase budget.')
                        self.curr_budgets = self.curr_budgets + \
                            self.increase_budget
                        if np.any(self.curr_budgets > self.break_vio_cost):
                            print('Can not find a feasible solution even \
                                  after the cost budget increase.')
                            print(f'Current budgets: {self.curr_budgets}.'
                                  + f'Break up budgets: {self.break_vio_cost}.'
                                  )
                            raise Exception('Can not find feasible points')
                        if np.any(self.curr_budgets > self.back_up_vio_cost):
                            print('Can not find a feasible solution even \
                                  after the cost budget increase.')
                            print(f'Current budgets: {self.curr_budgets}.'
                                  + f'Back up budgets: {self.back_up_vio_cost}.'
                                  )
                            # return a safe point
                            return self.x0_arr[0, :]

                        print('After increasing, current violation cost ' +
                              f'budget is {self.curr_budgets}')
            next_point_id = np.argmax(acq)
        elif acq_func_type == 'LCB':
            acq = self.get_acquisition(
                prob_eps=prob_eps, acq_func_type=acq_func_type)
            next_point_id = np.argmin(acq)
        else:
            raise Exception(f'{acq_func_type} acquistion not supported!')

        next_point = self.parameter_set[next_point_id]
        return next_point

    def make_step(self, update_gp=False, gp_package='gpy'):
        x_next, y_obj, constr_vals, vio_cost = self.step_sample_point(
            update_hyperparams=update_gp)
        vio_cost = np.squeeze(vio_cost)
        self.cumu_vio_cost = self.cumu_vio_cost + vio_cost
        self.curr_budgets = np.minimum(
            np.maximum(self.total_vio_budgets - self.cumu_vio_cost, 0),
            self.single_step_budget
        )
        self.curr_eval_budget -= 1
        return y_obj, constr_vals
