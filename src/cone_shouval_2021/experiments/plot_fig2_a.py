"""
Plots figure 2A, boxplot of reported times.
"""

import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
import importlib.util

from fna.tools.visualization.helper import set_global_rcParams
from fna.tools.utils import logger
from fna.tools.utils.data_handling import set_storage_locations

from helper import plotting
from helper import fig_defaults as fig_def
from utils.parameters import ParameterSet

logprint = logger.get_logger(__name__)

def plot_multi_batch(parameters, storage_paths, ax, plot_traces=True):
    """
    """
    filename = 'Recordings_{}.data'.format(parameters.label)

    try:
        with open(os.path.join(storage_paths['activity'], filename), 'rb') as f:
            data = pickle.load(f)

        reward_times = data['reward_times']
        recorded_data = data['recorded_data']

    except Exception as e:
        logprint.error(f'Exception occured: {str(e)}.\n\tSkipping dataset {parameters.label}')
        return

    colors = plotting.sns.color_palette('tab10')[:6]
    plotting.boxplot_multitrial(parameters, recorded_data, reward_times, ax, plot_rewards=False, colors=colors)

    test_batches = len(recorded_data['test_epoch'].keys())
    ax.set_title(f"Reported times, {test_batches} trials,\none learning instance")


def plot_multi_instance(parameters, storage_paths, ax, plot_traces=True):
    main_param_file = os.path.join(storage_paths['main'], 'ParameterSpace.py')
    spec = importlib.util.spec_from_file_location("ParameterSpace", main_param_file)
    mod_param = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod_param)

    trials = mod_param.ParameterRange['T']
    recorded_data = []
    for trial in trials:
        inst_label = parameters.label[:parameters.label.find('_T=')] + f'_T={trial}'
        filename = 'Recordings_{}.data'.format(inst_label)

        try:
            with open(os.path.join(storage_paths['activity'], filename), 'rb') as f:
                data = pickle.load(f)

            recorded_data.append(data['recorded_data'])

        except Exception as e:
            logprint.error(f'Exception occured: {str(e)}.\n\tSkipping dataset {filename}')
            continue

    plotting.boxplot_multi_instance(parameters, recorded_data, ax)


def run(parameters, display=False, plot=True, save=True, load_inputs=False):
    # experiments parameters
    if not isinstance(parameters, ParameterSet):
        parameters = ParameterSet(parameters)

    storage_paths = set_storage_locations(parameters.kernel_pars.data_path, parameters.kernel_pars.data_prefix,
                                          parameters.label, save=save)

    logger.update_log_handles(job_name=parameters.label, path=storage_paths['logs'])
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=fig_def.figsize['re_fig3s2_boxplot'], sharex=True)

    plot_multi_batch(parameters, storage_paths, axes[0])
    plot_multi_instance(parameters, storage_paths, axes[1])
    n_cols = parameters.net_pars.n_cols
    token_durations = np.array([sum(parameters.input_pars.misc.token_duration[0:i+1]) for i in range(n_cols)])

    ylim_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    for _ax in axes:
        _ax.grid(False)
        _ax.set_title(None)
        _ax.set_ylabel(None)
        _ax.set_ylim(0.,  ylim_max + 200)
        _ax.set_yticks(token_durations)
        # _ax.set_yticks([700., 1400., 2100., 2800.])
        _ax.set_xticklabels(list(range(n_cols)))
        _ax.set_xlabel('Column')
        _ax.yaxis.set_tick_params(labelbottom=True)

    axes[0].set_title('Recall times, \none instance')
    axes[1].set_title('Median recall times,\n10 instances')
    axes[0].set_ylabel('Time (s)')
    # axes[0].set_yticklabels([0.7, 1.4, 2.1, 2.8])
    axes[0].set_yticklabels(np.around(token_durations / 1000., decimals=1))
    axes[1].set_yticklabels([])

    fig.tight_layout()
    fig.savefig(os.path.join(storage_paths['figures'], 'Fig_2A_boxplot{}.pdf'.format(parameters.label)))
    fig.savefig(os.path.join(storage_paths['figures'], 'Fig_2A_boxplot{}.svg'.format(parameters.label)))
