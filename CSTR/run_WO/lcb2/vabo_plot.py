from matplotlib import pyplot as plt
import numpy as np

DIVIDE = 7


def plot_sampled_data(config_list, save_fig=False):
    legend_list = []
    i = 1
    for config in config_list:
        plt.plot(config['train_X'], config['obj'](config['train_X']))
        legend_list.append('sampled function ' + str(i))
        i += 1
    all_zeros = np.zeros_like(config['train_X'])
    plt.plot(config['train_X'], all_zeros, linestyle='--')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim((-5, 5))
    plt.legend(legend_list)
    if save_fig:
        plt.savefig('../figs/GP_1_dim_sampled_funcs.pdf', format='pdf')


def plot_gp_1d(gp, inputs, acq_feasible_points, beta=3, fmin=None,
               is_safe_opt=False, save_path=None, save_format='pdf', **kwargs):
    point_marker_linewidth = 3.5
    point_marker_size = 200
    figure = plt.figure()
    axis = figure.gca()

    mean, var = gp._raw_predict(inputs)

    output = mean.squeeze()
    std_dev = beta * np.sqrt(var.squeeze())
    axis.scatter(acq_feasible_points, np.zeros_like(acq_feasible_points),
                 marker='o', linewidth=5, alpha=0.3, color='g')

    if is_safe_opt:
        axis.fill_between(inputs[:, 0],
                          -(output - std_dev),
                          -(output + std_dev),
                          facecolor='blue',
                          alpha=0.3)

        axis.plot(inputs[:, 0], -output, **kwargs)
        axis.scatter(gp.X[:-1, 0], -gp.Y[:-1, 0], s=point_marker_size,
                     marker='x', linewidths=point_marker_linewidth, color='k')
        axis.scatter(gp.X[-1, 0], -gp.Y[-1, 0], s=point_marker_size,
                     marker='x', linewidths=point_marker_linewidth, color='r')
    else:
        axis.fill_between(inputs[:, 0],
                          output - std_dev,
                          output + std_dev,
                          facecolor='blue',
                          alpha=0.3)

        axis.plot(inputs[:, 0], output, **kwargs)
        axis.scatter(gp.X[:-1, 0], gp.Y[:-1, 0], s=point_marker_size,
                     marker='x', linewidths=point_marker_linewidth, color='k')
        axis.scatter(gp.X[-1, 0], gp.Y[-1, 0], s=point_marker_size,
                     marker='x', linewidths=point_marker_linewidth, color='r')

    axis.set_xlim([np.min(inputs[:, 0]),
                   np.max(inputs[:, 0])])

    if fmin is not None:
        axis.plot(inputs[[0, -1], 0], [fmin, fmin], 'r--')
    if save_path is not None:
        plt.savefig(save_path + '.' + save_format, format=save_format)

    return axis


def plot_results(safe_bo_total_cost_list, safe_bo_best_obj_list,
                 constrained_bo_total_cost_list, constrained_bo_best_obj_list,
                 violation_aware_bo_total_cost_list_set, violation_aware_bo_best_obj_list_set,
                 all_safe_cost_arr, all_safe_SR_arr,
                 all_con_bo_cost_arr, all_con_bo_SR_arr,
                 all_vabo_cost_arr, all_vabo_SR_arr,
                 with_error_bar=True):
    num_step, num_constr = safe_bo_total_cost_list.shape
    x = list(range(num_step))
    for i in range(num_constr):
        plt.figure()
        if with_error_bar:
            safe_cost_err = np.std(all_safe_cost_arr[:, :, i], axis=0) / DIVIDE
            plt.errorbar(x, safe_bo_total_cost_list[:, i], yerr=safe_cost_err, marker='*')

            constrained_cost_err = np.std(all_con_bo_cost_arr[:, :, i], axis=0) / DIVIDE
            plt.errorbar(x, constrained_bo_total_cost_list[:, i], yerr=constrained_cost_err, marker='o')
        else:
            plt.plot(np.array(safe_bo_total_cost_list)[:, i], marker='*')
            plt.plot(np.array(constrained_bo_total_cost_list)[:, i], marker='o')
        legends_list = ['Safe BO', 'Generic Constrained BO']

        i_tmp = 0
        for violation_aware_bo_total_cost_list, budget in violation_aware_bo_total_cost_list_set:

            if with_error_bar:
                vabo_cost_err = np.std(all_vabo_cost_arr[i_tmp, :, :, i], axis=0) / DIVIDE
                plt.errorbar(x, np.array(violation_aware_bo_total_cost_list)[:, i], yerr=vabo_cost_err, marker='+')
            else:
                plt.plot(np.array(violation_aware_bo_total_cost_list)[:, i], marker='+')
            i_tmp += 1
            legends_list.append('Violation Aware BO ' + str(budget))
            # print(violation_aware_bo_total_cost_list)

        # plt.plot(np.array([vio_budget]*(optimization_config['eval_budget']+1)), marker='v', color='r')
        plt.xlim((0, num_step))
        plt.ylim((0, 12))
        print(legends_list)
        plt.legend(legends_list)
        plt.xlabel('Step')
        plt.ylabel('Cumulative cost')
        plt.savefig('../figs/cost_inc' + '_constr_' + str(i) + '.pdf', format='pdf')

    # compare convergence
    plt.figure()
    if with_error_bar:
        safe_bo_obj_err = np.std(all_safe_SR_arr, axis=0) / DIVIDE
        plt.errorbar(x, safe_bo_best_obj_list, yerr=safe_bo_obj_err, marker='*')
        constrained_bo_obj_err = np.std(all_con_bo_SR_arr, axis=0)
        plt.errorbar(x, constrained_bo_best_obj_list, yerr=constrained_bo_obj_err, marker='o')
    else:
        plt.plot(safe_bo_best_obj_list, marker='*')
        plt.plot(constrained_bo_best_obj_list, marker='o')

    i_tmp = 0
    for violation_aware_bo_best_obj_list, budget in violation_aware_bo_best_obj_list_set:
        if with_error_bar:
            vabo_obj_err = np.std(all_vabo_SR_arr[i_tmp, :, :], axis=0) / DIVIDE
            plt.errorbar(x, violation_aware_bo_best_obj_list, yerr=vabo_obj_err, marker='+')
        else:
            plt.plot(violation_aware_bo_best_obj_list, marker='+')
        i_tmp += 1

    plt.xlim((0, num_step))
    plt.legend(legends_list)
    plt.xlabel('Step')
    plt.ylabel('Simple Regret')
    plt.savefig('../figs/best_obj.pdf', format='pdf')


def cdf_plot(data, bins=100, hist_range=(0, 50)):
    count, bins_count = np.histogram(data, bins=bins, range=hist_range)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], cdf, label="CDF")


def plot_cost_SR_scatter(cost_lists, SR_lists, fig_name=None):
    cost_arr = np.array(cost_lists)
    num_tra, num_eval, num_constr = cost_arr.shape

    SR_arr = np.array(SR_lists)

    for i in range(num_constr):
        plt.figure()
        plt.scatter(cost_arr[:, -1, i], SR_arr[:, -1])
        plt.xlabel('Violation cost ' + str(i + 1))
        plt.ylabel('Simple regret')
        plt.title(fig_name)

    if fig_name is not None:
        plt.savefig('../figs/' + fig_name, format='png')
