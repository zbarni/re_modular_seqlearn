"""
Plots single-trial rates.
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


def plot_single_psth(parameters, reward_times, recorded_data, storage_paths, plot_epoch, plot_batch,
                     plot_labels=True, ymax=80., plot_size='small', train_set=False, savefig=False, inset=False):
    """
    """
    stim_width = 10 if ymax > 60 else 3
    if plot_size == 'small':
        fr_y_lim_max = ymax + 10
        yticks = [0., int(ymax/2), int(ymax)]
        time_x_lim_max = 3000.
        fig, ax = plt.subplots(figsize=fig_def.figsize['psth_single'])
    else:
        fr_y_lim_max = ymax
        yticks = [0., int(ymax/2), int(ymax)]
        time_x_lim_max = 1000. if not inset else 1800.
        fig, ax = plt.subplots(figsize=fig_def.figsize['psth_single_large'])

    colors = plotting.sns.color_palette('tab10')[:4]
    if str.isdigit(plot_epoch):
        plot_epoch = str(plot_epoch)
        plot_batch = str(plot_batch)

    # labels = {c: [r"T $(C_{" + str(c) + "})$",
    #               r"M $(C_{" + str(c) + "})$",
    #               r"I $(L_5, C_{" + str(c) + "})$",
    #               r"I $(L_{23}, C_{" + str(c) + "})$",
    #               ] for c in range(2)}

    # labels = {c: [r"$\mathrm{T}_{" + str(c) + "}$",
    #               r"$\mathrm{M}_{" + str(c) + "}$",
    #               r"$\mathrm{I}_{" + str(c) + "} (L_5)$",
    #               r"$\mathrm{I}_{" + str(c) + "} (L_{23})$",
    #               ] for c in range(2)}
    labels = {c: [r"$\mathrm{T}^{" + str(c + 1) + "}$",
                  r"$\mathrm{M}^{" + str(c + 1) + "}$",
                  r"$\mathrm{I}^{" + str(c + 1) + "}_\mathrm{T}$",
                  r"$\mathrm{I}^{" + str(c + 1) + "}_\mathrm{M}$",
                  ] for c in range(1)}
    labels[1] = [None for _ in range(4)]  # don't plot C2

    plotting.plot_single_psth(parameters, recorded_data, reward_times, plot_epoch, plot_batch, ax, colors=colors,
                              linewidths = [1.0 for _ in range(4)], plot_inh_l23=True, plot_rewards=True,
                              plot_labels=labels)

    token_durations = parameters.input_pars.misc.token_duration

    if plot_labels:
        ax.set_ylabel('Firing rate\n(spks/sec)', fontsize=11)
        ax.set_xlabel('Time (ms)', fontsize=11)

    ax.grid(False)
    ax.set_title(None)
    ax.set_ylim(0., fr_y_lim_max)
    ax.set_yticks(yticks)
    ax.set_xticks([sum(token_durations[:idx]) for idx in range(len(token_durations))])
    ax.set_xlim(0., time_x_lim_max)
    ax.margins(x=0.)
    # plt.legend(loc='center right', fontsize=5)
    if plot_size == 'small':
        plt.legend(loc='upper right', fontsize=6)
    else:
        plt.legend(loc='upper right', fontsize=6,  framealpha=1.)

    # plot stimulus bars
    if train_set:
        for idx in range(2):
            xmin = np.sum(token_durations[:idx]) / time_x_lim_max
            xmax = np.sum(token_durations[:idx + 1]) / time_x_lim_max
            ax.axhspan(fr_y_lim_max - stim_width, fr_y_lim_max, xmin=xmin, xmax=xmax, facecolor=colors[idx], alpha=0.3)
    else:
        ax.axhspan(fr_y_lim_max - 10, fr_y_lim_max, xmin=0., xmax=token_durations[0] / time_x_lim_max,
                        facecolor=colors[0], alpha=0.3)

    fig.tight_layout()
    filename = f"Fig3_singleInstance_{parameters.label}_epoch={plot_epoch}_batch={plot_batch}_ymax={ymax}"
    if savefig:
        fig.savefig(os.path.join(storage_paths['figures'], filename + '.pdf'))
        fig.savefig(os.path.join(storage_paths['figures'], filename + '.svg'), format='svg')
        print(f"Finished plotting figure for {parameters.label}")


def run(parameters, display=False, plot=True, save=True, plot_epoch=None, plot_batch=None, plot_labels=True,
        ymax=80., train_set=False, plot_size='small'):
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
                     plot_size=plot_size)

    processed = True