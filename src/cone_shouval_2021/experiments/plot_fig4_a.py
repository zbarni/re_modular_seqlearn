"""
Plots the median recall time as a function of a certain weights.

Figure 4A.
"""
import pickle
import os
import numpy as np
from matplotlib import pyplot as plt

from fna.tools.utils import logger
from fna.tools.utils.data_handling import set_storage_locations

from helper import plotting
from helper import fig_defaults as fig_def
from utils.parameters import ParameterSet
from multiprocessing.dummy import Pool as ThreadPool


logprint = logger.get_logger(__name__)
processed = False


def calc_median_and_outliers(parameters, storage_paths):
    """
    """
    filename = 'Recordings_{}.data'.format(parameters.label)

    try:
        with open(os.path.join(storage_paths['activity'], filename), 'rb') as f:
            data = pickle.load(f)

        recorded_data = data['recorded_data']

    except Exception as e:
        logprint.error(f'Exception occured: {str(e)}.\n\tSkipping dataset {parameters.label}')
        return None

    reported_times = plotting.__get_reported_times(parameters, recorded_data, 'test_epoch', ['timer_exc'])
    onset_times = np.cumsum([0] + parameters.input_pars.misc.token_duration[:-1])
    reported_times = [np.array(x) - onset_times[idx] for idx, x in enumerate(reported_times.values())]

    medians = [np.median(x) for x in reported_times]
    logprint.info(f"Medians from dataset with {parameters.label}: {medians}")
    return medians


def plot_recalltime_vs_weights(parameters, storage_paths):
    """

    :param parameters:
    :param storage_paths:
    :return:
    """
    main_label = storage_paths['label']

    param_range = {
        'w_IE_L5': np.arange(-25., 41., 5.),
        'T': np.arange(1, 6, 1)
    }

    def __process(w):
        """

        :param w:
        :return:
        """
        tmp_medians = [[] for _ in range(3)]
        for trial in param_range['T']:
            filename = f'{main_label}_w_IE_L5={w}_T={trial}'
            # logprint.info(f"Processing dataset: {filename}")
            abs_path = str(os.path.join(storage_paths['parameters'], filename))
            try:
                parameters_trial = ParameterSet(abs_path)
            except:
                # logprint.error(f"Could not open dataset: {abs_path}")
                continue

            res_medians = calc_median_and_outliers(parameters_trial, storage_paths)
            if res_medians:
                for col_ in range(3):
                    tmp_medians[col_].append(res_medians[col_])

        res = [(np.mean(tmp_medians[col_]), np.std(tmp_medians[col_])) for col_ in range(3)]
        return res
    # end __process()

    thread_args = list(param_range['w_IE_L5'])
    pool = ThreadPool(len(thread_args))
    results = pool.map(__process, thread_args)
    pool.close()
    pool.join()

    # # plot median and variance over all columns
    fig, ax = plt.subplots(figsize=fig_def.figsize['median_vs_weights'])
    colors = plotting.sns.color_palette('tab10')[:4]

    for col in range(3):
        # mean
        ax.plot(param_range['w_IE_L5'], [x[col][0] - 700 for x in results], color=colors[col],
                label=r'$\mathrm{C}_{' + str(col) + '}$')
                # label=r'$\Delta w \ T \rightarrow I_\mathrm{T} \ (20\%)$')

    ax.plot(param_range['w_IE_L5'], [0] * len(param_range['w_IE_L5']), '--', color='k')

    ax.set_xlim([-50, 50])
    ax.axvspan(-50, -25, facecolor='grey', alpha=0.3)
    ax.set_yticks([-200, 0, +200])
    ax.set_yticklabels(['-200', '0', '+200'])
    ax.set_ylabel("Median\nrecall time")
    # ax.set_xlabel(r"$\mathrm{w}_{IE}^{TT}$ (%)")
    ax.set_xlabel(r"$\Delta w \ T \rightarrow I_\mathrm{T} \ (\%)$")
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=False, labeltop=True)
    ax.xaxis.set_label_position('top')
    ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(os.path.join(storage_paths['figures'], f'Fig_median_vs_weights_{parameters.label}.pdf'))
    fig.savefig(os.path.join(storage_paths['figures'], f'Fig_median_vs_weights_{parameters.label}.svg'))
    fig.savefig(os.path.join(storage_paths['figures'], f'Fig_median_vs_weights_{parameters.label}.eps'), format='eps')


def run(parameters, display=False, plot=True, save=True):
    global processed
    if processed:
        exit(-1)

    # experiments parameters
    if not isinstance(parameters, ParameterSet):
        parameters = ParameterSet(parameters)

    storage_paths = set_storage_locations(parameters.kernel_pars.data_path, parameters.kernel_pars.data_prefix,
                                          parameters.label, save=save)

    logger.update_log_handles(job_name=parameters.label, path=storage_paths['logs'])

    plot_recalltime_vs_weights(parameters, storage_paths)
    processed = True