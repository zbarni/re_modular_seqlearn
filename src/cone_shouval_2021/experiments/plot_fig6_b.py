"""
Plots figure 6B
"""
import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from fna.tools.visualization.helper import set_global_rcParams
from fna.tools.utils import logger
from fna.tools.utils.data_handling import set_storage_locations

from helper import plotting
from helper import fig_defaults as fig_def
from utils.parameters import ParameterSet

logprint = logger.get_logger(__name__)


def plot_fig3_ff(parameters, reward_times, recorded_data, storage_paths,
                 plot_traces=True, savefig=False):
    """
    Plot Fig. 3 from eLife version in the case of all-to-all FF connections.
    """
    plot_epoch_batches = [
        ('transient_epoch', 'transient_batch'),
        # T=2
        ('0', '0'),
        ('0', '96'),
        ('test_epoch', 'test_batch_5')
    ]

    fig, axes = plt.subplots(nrows=4, figsize=fig_def.figsize['re_fig3_tall'], sharey=True)
    colors = plotting.sns.color_palette('tab10')[:4]
    plotting.plot_psth(parameters, recorded_data, reward_times, plot_epoch_batches, axes, plot_rewards=False,
                       colors=colors)

    fr_y_lim_max = 130.
    token_durations = parameters.input_pars.misc.token_duration
    for _ax in axes:
        _ax.grid(False)
        _ax.set_title(None)
        _ax.set_ylabel(None)
        _ax.set_ylim(0., fr_y_lim_max)
        _ax.set_yticks([0., 40., 80., 120.])
        _ax.set_xticks([sum(token_durations[:idx]) for idx in range(len(token_durations))])
        _ax.margins(x=0.)

    for _ax in axes[:-1]:
        _ax.set_xticklabels([])

    # plot stimulus bars
    axes[0].axhspan(fr_y_lim_max - 10, fr_y_lim_max, xmin=0., xmax=token_durations[0] / sum(token_durations),
                    facecolor=colors[0], alpha=0.3)
    axes[3].axhspan(fr_y_lim_max - 10, fr_y_lim_max, xmin=0., xmax=token_durations[0] / sum(token_durations),
                    facecolor=colors[0], alpha=0.3)
    prev_token_sum = 0.
    for idx, token in enumerate(token_durations[:-1]):
        axes[1].axhspan(fr_y_lim_max - 10, fr_y_lim_max, xmin=prev_token_sum / sum(token_durations),
                        xmax=(prev_token_sum + token) / sum(token_durations),
                        facecolor=colors[idx], alpha=0.3)
        axes[2].axhspan(fr_y_lim_max - 10, fr_y_lim_max, xmin=prev_token_sum / sum(token_durations),
                        xmax=(prev_token_sum + token) / sum(token_durations),
                        facecolor=colors[idx], alpha=0.3)
        prev_token_sum += token

    fig.supylabel('Firing rate (spks/sec)', fontsize=11)
    fig.supxlabel('Time (ms)', fontsize=11)
    fig.tight_layout()
    plt.subplots_adjust(left=0.18, right=0.97, bottom=0.13)
    if savefig:
        fig.savefig(os.path.join(storage_paths['figures'], 'Fig3_{}.pdf'.format(parameters.label)))
        fig.savefig(os.path.join(storage_paths['figures'], 'Fig3_{}.svg'.format(parameters.label)), format='svg')
        print(f"Finished plotting Figure 3 for {parameters.label}")


def run(parameters, display=False, plot=True, save=True, load_inputs=False, T=None, feedforward=False):
    """

    :param parameters:
    :param display:
    :param plot:
    :param save:
    :param load_inputs:
    :param T: (int) trial (experiment) number
    :param feedforward: (bool) whether it's the FF scenario
    :return:
    """
    # experiments parameters
    if not isinstance(parameters, ParameterSet):
        parameters = ParameterSet(parameters)

    if T:
        if 'T=' + str(T) not in parameters.label:
            logprint.info(f"Skipping dataset {parameters.label}...")
            return
        else:
            logprint.info(f"Found trial T={T}, now processing...")
    else:
        logprint.info(f"Processing all trials...")

    storage_paths = set_storage_locations(parameters.kernel_pars.data_path, parameters.kernel_pars.data_prefix,
                                          parameters.label, save=save)

    logger.update_log_handles(job_name=parameters.label, path=storage_paths['logs'])
    filename = 'Recordings_{}.data'.format(parameters.label)

    try:
        with open(os.path.join(storage_paths['activity'], filename), 'rb') as f:
            data = pickle.load(f)

        reward_times = data['reward_times']
        recorded_data = data['recorded_data']

        plot_fig3_ff(parameters, reward_times, recorded_data, storage_paths, savefig=True, plot_traces=True)
    except Exception as e:
        print('Exception occured!', str(e))