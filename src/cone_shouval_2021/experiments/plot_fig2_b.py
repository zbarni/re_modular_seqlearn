"""
Plots figure 2B, weights.
"""
import os
from matplotlib import pyplot as plt

from fna.tools.utils import logger
from fna.tools.utils.data_handling import set_storage_locations

from helper import plotting
from helper import fig_defaults as fig_def
from utils.parameters import ParameterSet

logprint = logger.get_logger(__name__)

def plot_weights_sum(parameters, storage_paths):
    """
    Plot Fig. 3.S2.b) from eLife version.
    """
    plot_cols = [0, 1, 2, 3]
    colors = plotting.sns.color_palette('tab10')[:4]

    filename = 'Recordings_{}.data'.format(parameters.label)
    fig, axes = plt.subplots(2, 1, figsize=fig_def.figsize['re_fig3s2_weights'], sharex=True)

    reward_times, recorded_data = plotting.load_recording_data(filename, storage_paths)
    data = plotting.get_weights(parameters, recorded_data, mean=False)  # TODO change this when new data available

    for col_id in plot_cols:
        axes[1].plot(data['trial'][col_id], data['recurrent'][col_id], '-', color=colors[col_id],
                     label=r'$\mathrm{C}_' + str(col_id) + '$')
        try:
            axes[0].plot(data['trial'][col_id], data['feedforward'][col_id], '-', color=colors[col_id],
                         label=r'$\mathrm{C}_' + str(col_id) + ' \\rightarrow ' + str(col_id + 1) + '$')
        except:
            pass

    for _ax in axes:
        _ax.grid(False)
        _ax.set_title(None)
        _ax.set_ylabel(None)
        # _ax.minorticks_on()

        _ax.set_xlabel('Trial')
        # _ax.xaxis.set_tick_params(width=2, length=6)
        _ax.xaxis.set_tick_params(labelbottom=True)
        _ax.legend(loc='best')

    axes[0].set_title('Sum of feedforward weights')
    axes[0].set_ylabel('nS')
    axes[1].set_title('Sum of recurrent weights')
    axes[1].set_ylabel('nS')

    fig.tight_layout()
    fig.savefig(os.path.join(storage_paths['figures'], 'Fig3S2_weights_sum_{}.pdf'.format(parameters.label)))


def plot_weights_mean(parameters, storage_paths):
    """
    Plot Fig. 3.S2.b) from eLife version.
    """
    plot_cols = [0, 1, 2, 3]
    colors = plotting.sns.color_palette('tab10')[:4]

    filename = 'Recordings_{}.data'.format(parameters.label)
    fig, axes = plt.subplots(2, 1, figsize=fig_def.figsize['re_fig3s2_weights'], sharex=True)

    reward_times, recorded_data = plotting.load_recording_data(filename, storage_paths)
    data = plotting.get_weights(parameters, recorded_data, mean=True)

    for col_id in plot_cols:
        axes[1].plot(data['trial'][col_id], data['recurrent'][col_id], '-', color=colors[col_id],
                     label=r'$\mathrm{C}_' + str(col_id) + '$')
        try:
            axes[0].plot(data['trial'][col_id], data['feedforward'][col_id], '-', color=colors[col_id],
                         label=r'$\mathrm{C}_' + str(col_id) + ' \\rightarrow ' + '\mathrm{C}_' + str(col_id + 1) + '$')
        except:
            pass

    for idx, _ax in enumerate(axes):
        _ax.grid(False)
        _ax.set_title(None)
        _ax.set_ylabel(None)
        _ax.xaxis.set_tick_params(labelbottom=True)
        _ax.legend(loc=['best', 'upper right'][idx])

    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-1, 1))
    # axes[1].yaxis.set_major_formatter(formatter)

    axes[0].set_title('Mean feedforward weights')
    axes[0].set_ylabel('nS')
    axes[1].set_title('Mean recurrent weights')
    axes[1].set_xlabel('Trial')
    axes[1].set_ylabel('nS')

    fig.tight_layout()
    fig.savefig(os.path.join(storage_paths['figures'], 'Fig3S2_weights_mean_{}.pdf'.format(parameters.label)))


def run(parameters, display=False, plot=True, save=True, load_inputs=False):
    # ############################ SYSTEM
    # experiments parameters
    if not isinstance(parameters, ParameterSet):
        parameters = ParameterSet(parameters)

    # if 'T=8' not in parameters.label:
    #     logprint.warning(f"For now, we only print Trial 8 from `seq_replay_original_4x700`")
    #     return

    storage_paths = set_storage_locations(parameters.kernel_pars.data_path, parameters.kernel_pars.data_prefix,
                                          parameters.label, save=save)

    logger.update_log_handles(job_name=parameters.label, path=storage_paths['logs'])


    plot_weights_sum(parameters, storage_paths)
    plot_weights_mean(parameters, storage_paths)


