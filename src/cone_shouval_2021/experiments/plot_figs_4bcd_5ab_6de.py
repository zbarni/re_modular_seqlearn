"""
Plots single-trial rates.
Used for figures 4B-D, 5A,B and 6D,E
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
processed = False


def plot_single_psth(parameters, reward_times, recorded_data, storage_paths, plot_epoch, plot_batch, plot_labels=True,
                     ymax=80., plot_size='small', time_xlim_max=None, train_set=False, savefig=False):
    """
    """
    if plot_size == 'small':
        fr_y_lim_max = ymax + 10
        yticks = [0., int(ymax/2), int(ymax)]
        tmp_time_xlim_max = 3000.
        fig, ax = plt.subplots(figsize=fig_def.figsize['psth_single'])
    else:
        fr_y_lim_max = ymax
        yticks = [0., int(ymax/2), int(ymax)]
        tmp_time_xlim_max = 2500.
        fig, ax = plt.subplots(figsize=fig_def.figsize['psth_single_large'])

    time_xlim_max = tmp_time_xlim_max if time_xlim_max is None else float(time_xlim_max)
    print("time_xlim_max", time_xlim_max)

    colors = plotting.sns.color_palette('tab10')[:4]
    if str.isdigit(plot_epoch):
        plot_epoch = str(plot_epoch)
        plot_batch = str(plot_batch)

    plotting.plot_single_psth(parameters, recorded_data, reward_times, plot_epoch, plot_batch, ax, colors=colors,
                              plot_rewards=False)

    token_durations = parameters.input_pars.misc.token_duration

    if plot_labels:
        ax.set_ylabel('Firing rate\n(spks/sec)', fontsize=11)
        ax.set_xlabel('Time (ms)', fontsize=11)

    ax.grid(False)
    ax.set_title(None)
    ax.set_ylim(0., fr_y_lim_max)
    ax.set_yticks(yticks)
    ax.set_xticks([sum(token_durations[:idx]) for idx in range(len(token_durations))])
    ax.set_xlim(0., time_xlim_max)
    ax.margins(x=0.)

    # plot stimulus bars
    if train_set:
        for idx in range(3):
            xmin = np.sum(token_durations[:idx]) / time_xlim_max
            xmax = np.sum(token_durations[:idx + 1]) / time_xlim_max
            print(xmax, xmin)
            ax.axhspan(fr_y_lim_max - 10, fr_y_lim_max, xmin=xmin, xmax=xmax, facecolor=colors[idx], alpha=0.3)
    else:
        ax.axhspan(fr_y_lim_max - 10, fr_y_lim_max, xmin=0., xmax=token_durations[0] / time_xlim_max,
                   facecolor=colors[0], alpha=0.3)

    fig.tight_layout()
    filename = f"Fig3_singleInstance_{parameters.label}_epoch={plot_epoch}_batch={plot_batch}"
    if savefig:
        fig.savefig(os.path.join(storage_paths['figures'], filename + '.pdf'))
        fig.savefig(os.path.join(storage_paths['figures'], filename + '.svg'), format='svg')
        print(f"Finished plotting figure for {parameters.label}")


def run(parameters, display=False, plot=True, save=True, plot_epoch=None, plot_batch=None, plot_labels=True,
        ymax=80., train_set=False, plot_size='small', time_xlim_max=None):
    """

    :param parameters:
    :param display:
    :param plot:
    :param save:
    :param plot_epoch: which epoch to plot
    :param plot_batch: which batch to plot
    :param plot_labels:
    :param ymax:
    :return:
    """
    global processed
    if processed:
        exit(-1)

    if not isinstance(parameters, ParameterSet):
        parameters = ParameterSet(parameters)

    storage_paths = set_storage_locations(parameters.kernel_pars.data_path, parameters.kernel_pars.data_prefix,
                                          parameters.label, save=save)

    logger.update_log_handles(job_name=parameters.label, path=storage_paths['logs'])
    filename = 'Recordings_{}.data'.format(parameters.label)

    try:
        with open(os.path.join(storage_paths['activity'], filename), 'rb') as f:
            data = pickle.load(f)

    except Exception as e:
        print('Exception occured!', str(e), "Maybe try a different epoch / batch?")

    reward_times = data['reward_times']
    recorded_data = data['recorded_data']

    plot_single_psth(parameters, reward_times, recorded_data, storage_paths, savefig=True, train_set=train_set,
                     plot_epoch=plot_epoch, plot_batch=plot_batch, plot_labels=plot_labels, ymax=float(ymax),
                     plot_size=plot_size, time_xlim_max=time_xlim_max)

    processed = True