"""
Plots figure 3A and 3B left column, robustness of the CNA circuit to variation in the weights.
"""
import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import importlib.util

from fna.tools.visualization.helper import set_global_rcParams
from fna.tools.utils import logger
from fna.tools.utils.data_handling import set_storage_locations

from helper import plotting
from helper import fig_defaults as fig_def
from utils.parameters import ParameterSet
from multiprocessing.dummy import Pool as ThreadPool


logprint = logger.get_logger(__name__)
processed = False

labelsize = 10
ticksize = 9

def calc_median_and_outliers(parameters, storage_paths, manual_outl_th=False):
    """

    :param parameters:
    :param storage_paths:
    :param manual_outl_th:
    :return:
    """
    filename = 'Recordings_{}.data'.format(parameters.label)

    try:
        with open(os.path.join(storage_paths['activity'], filename), 'rb') as f:
            data = pickle.load(f)

        recorded_data = data['recorded_data']

    except Exception as e:
        logprint.error(f'Exception occured: {str(e)}.\n\tSkipping dataset {parameters.label}')
        return None, None

    reported_times = plotting.__get_reported_times(parameters, recorded_data, 'test_epoch', ['timer_exc'])
    onset_times = np.cumsum([0] + parameters.input_pars.misc.token_duration[:-1])
    reported_times = [np.array(x) - onset_times[idx] for idx, x in enumerate(reported_times.values())]

    outliers_mad = [np.count_nonzero(plotting.is_outlier(x, thresh=3)) for x in reported_times]
    logprint.info(f"{parameters.label} outliers: {outliers_mad}")
    # manually compute outliers
    outliers_abs = []
    for idx, rt in enumerate(reported_times):
        token_dur = parameters.input_pars.misc.token_duration[idx]
        outlier_th = token_dur * 0.2  # 20%
        outliers_abs.append(len(np.where(~((rt >= token_dur - outlier_th) & (rt <= token_dur + outlier_th)))[0]))

        # outlier_idxs = np.where(~((rt >= token_dur - outlier_th) & (rt <= token_dur + outlier_th)))[0]
        # print(f"Outliers: {rt[outlier_idxs][:10]}, indices {outlier_idxs[:10]}")
        # print(f"Minimum: {np.sort(rt)[:10]}\n")
        # print(f"{filename}: #outliers {outliers[-1]}")

    medians = [np.median(x) for x in reported_times]
    return medians, outliers_abs, outliers_mad


def plot_cna_robustness(parameters, storage_paths):

    param_range = {
        'w_EE_MT': [-20, 0., 20.0],
        'w_EI_MT': [-20.0, 0., 20.0],
        'T': np.arange(1, 21, 1)
        # 'T': np.arange(1, 2, 1)
    }

    main_label = storage_paths['label']

    outliers_abs = [np.zeros((3,3)) for _ in range(4)]
    outliers_mad = [np.zeros((3,3)) for _ in range(4)]
    medians = [np.zeros((3,3)) for _ in range(4)]

    outliers_col_abs_mean = np.zeros((3,3))
    outliers_col_mad_mean = np.zeros((3,3))
    medians_col_mean = np.zeros((3,3))

    for i in range(3):
        for j in range(3):
            logprint.info(f'Processing parameter combination w_EE_MT={param_range["w_EE_MT"][i]}, '
                          f'w_EI_MT={param_range["w_EI_MT"][j]}...')
            def __process_trial(T):
                """

                :param T:
                :return:
                """
                filename = f'{main_label}_w_EE_MT={param_range["w_EE_MT"][i]}_w_EI_MT={param_range["w_EI_MT"][j]}_T={T}'
                logprint.info(f"Processing dataset: {filename}")
                abs_path = str(os.path.join(storage_paths['parameters'], filename))
                try:
                    parameters_trial = ParameterSet(abs_path)
                except:
                    logprint.error(f"Could not open dataset: {abs_path}")
                    return None, None
                return calc_median_and_outliers(parameters_trial, storage_paths)
            # end __process_trial()

            thread_args = list(param_range['T'])
            pool = ThreadPool(len(thread_args))
            results = pool.map(__process_trial, thread_args)
            pool.close()
            pool.join()

            tmp_medians = [[] for _ in range(4)]
            tmp_outliers_abs = [[] for _ in range(4)]
            tmp_outliers_mad = [[] for _ in range(4)]

            # gather and restructure results from multithreading
            for idx in range(len(param_range['T'])):
                if results[idx][0]:
                    for col in range(4):
                        tmp_medians[col].append(results[idx][0][col])
                        tmp_outliers_abs[col].append(results[idx][1][col])
                        tmp_outliers_mad[col].append(results[idx][2][col])

            for col in range(4):
                outliers_abs[col][i][j] = np.mean(tmp_outliers_abs[col])
                outliers_mad[col][i][j] = np.mean(tmp_outliers_mad[col])
                medians[col][i][j] = np.mean(tmp_medians[col])

            # compute global mean over all columns/modules
            outliers_col_abs_mean[i][j] = np.mean(np.array(tmp_outliers_abs))
            outliers_col_mad_mean[i][j] = np.mean(np.array(tmp_outliers_mad))
            medians_col_mean[i][j] = np.mean(np.array(tmp_medians))

    ### plot outliers
    single_col = 3
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=fig_def.figsize['re_fig5s1_scan'])

    plotting.sns.heatmap(np.flipud(outliers_abs[single_col]), ax=axes[0][0], cbar=True,
                         cmap=plotting.sns.color_palette("flare", as_cmap=True), cbar_kws={'ticks':[16, 20]})
    axes[0][0].figure.axes[-1].tick_params(labelsize=7)
    plotting.sns.heatmap(np.flipud(outliers_mad[single_col]), ax=axes[0][1], cbar=True,
                         cmap=plotting.sns.color_palette("flare", as_cmap=True), cbar_kws={'ticks':[0.5, 1.]})
    axes[0][1].figure.axes[-1].tick_params(labelsize=7)

    plotting.sns.heatmap(np.flipud(outliers_col_abs_mean), ax=axes[1][0], cbar=True,
                         cmap=plotting.sns.color_palette("flare", as_cmap=True), cbar_kws={'ticks':[12, 14]})
    axes[1][0].figure.axes[-1].tick_params(labelsize=7)
    plotting.sns.heatmap(np.flipud(outliers_col_mad_mean), ax=axes[1][1], cbar=True,
                         cmap=plotting.sns.color_palette("flare", as_cmap=True), cbar_kws={'ticks':[0.4, 0.6]})
    axes[1][1].figure.axes[-1].tick_params(labelsize=7)

    print("outliers_abs", np.flipud(outliers_abs[single_col]))
    print("outliers_mad", np.flipud(outliers_mad[single_col]))
    print("outliers_col_abs_mean", np.flipud(outliers_col_abs_mean))
    print("outliers_col_mad_mean", np.flipud(outliers_col_mad_mean))

    for _ax in axes.flatten():
        if _ax == axes[0][0]:
            _ax.set_ylabel(r"$T \rightarrow M$", fontsize=labelsize)
            _ax.set_xlabel(r"$I_\mathrm{T} \rightarrow M$ ", fontsize=labelsize)

        _ax.set_yticklabels(np.array(param_range['w_EE_MT'][::-1]).astype(int), fontsize=ticksize)
        _ax.set_xticklabels(np.array(param_range['w_EI_MT']).astype(int), fontsize=ticksize)

    fig.tight_layout()
    fig.savefig(os.path.join(storage_paths['figures'], f'Fig_3A_outliers_grid_M={col}.pdf'))
    fig.savefig(os.path.join(storage_paths['figures'], f'Fig_3A_outliers_grid_M={col}.svg'))

    vmin, vmax = -105, 55
    # # plot median and variance over all columns
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=fig_def.figsize['re_fig5s1_scan_med+std'])
    plotting.sns.heatmap(np.flipud(medians[col]) - 700., ax=axes[0], cbar=True,
                         vmin=vmin, vmax=vmax, cbar_kws={'ticks': [-100, 0, 50]},
                         cmap=plotting.sns.color_palette("flare", as_cmap=True))
    axes[0].figure.axes[-1].set_yticklabels(['-100', '0', '+50'])
    axes[0].figure.axes[-1].tick_params(labelsize=7)
    plotting.sns.heatmap(np.flipud(medians_col_mean) - 700., ax=axes[1], cbar=True,
                         vmin=vmin, vmax=vmax, cbar_kws={'ticks': [-100, 0, 50]},
                         cmap=plotting.sns.color_palette("flare", as_cmap=True))
    axes[1].figure.axes[-1].set_yticklabels(['-100', '0', '+50'])
    axes[1].figure.axes[-1].tick_params(labelsize=7)

    for _ax in axes:
        if _ax == axes[0]:
            _ax.set_ylabel(r"$T \rightarrow M$", fontsize=labelsize)
            _ax.set_xlabel(r"$I_\mathrm{T} \rightarrow M$ ", fontsize=labelsize)
        _ax.set_yticklabels(np.array(param_range['w_EE_MT'][::-1]).astype(int), fontsize=ticksize)
        _ax.set_xticklabels(np.array(param_range['w_EI_MT']).astype(int), fontsize=ticksize)

    fig.tight_layout()
    fig.savefig(os.path.join(storage_paths['figures'], f'Fig_3A_outliers_grid_medians.pdf'))
    fig.savefig(os.path.join(storage_paths['figures'], f'Fig_3A_outliers_grid_medians.svg'))


def run(parameters, display=False, plot=True, save=True, **extra):
    global processed
    if processed:
        exit(-1)
    # ############################ SYSTEM
    # experiments parameters
    if not isinstance(parameters, ParameterSet):
        parameters = ParameterSet(parameters)

    storage_paths = set_storage_locations(parameters.kernel_pars.data_path, parameters.kernel_pars.data_prefix,
                                          parameters.label, save=save)

    logger.update_log_handles(job_name=parameters.label, path=storage_paths['logs'])

    plot_cna_robustness(parameters, storage_paths)
    processed = True