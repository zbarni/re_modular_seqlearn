"""

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

logprint = logger.get_logger(__name__)


def plot_weights_mean(parameters, storage_paths, src_col_ff):
    """
    Plot Fig. 3.S2.b) from eLife version.
    """
    plot_cols = [0, 1, 2]
    # plot_cols = [0, 1]
    colors = plotting.sns.color_palette('tab10')[:4]

    filename = 'Recordings_{}.data'.format(parameters.label)
    # fig, axes = plt.subplots(1, 2, figsize=fig_def.figsize['re_fig3s2_weights_landscape'], sharex=True)
    fig, axes = plt.subplots(2, 1, figsize=fig_def.figsize['re_fig3s2_weights_landscape'], sharex=True)

    try:
        reward_times, recorded_data = plotting.load_recording_data(filename, storage_paths)
    except:
        logprint.error(f"Skipping dataset {parameters.label}...")
        return
    data = plotting.get_weights_ff_all(parameters, recorded_data, mean=True, source_col=src_col_ff)

    for col_id in plot_cols:
        axes[1].plot(data['trial'][col_id], data['recurrent'][col_id], '-', color=colors[col_id],
                     label=r'$\mathrm{C}_' + str(col_id) + '$')
        try:
            axes[0].plot(data['trial'][col_id], data['feedforward'][col_id], '-', color=colors[col_id],
                         label=r'$\mathrm{C}_' + str(src_col_ff + 1) + ' \\rightarrow ' + '\mathrm{C}_' + str(col_id + 1) + '$')
            logprint.info(f"Plotted data for connection C{src_col_ff + 1} -> C{col_id + 1}")
        except Exception as e:
            logprint.warning(f"Couldn't plot data for connection C{src_col_ff + 1} -> C{col_id + 1}: {str(e)}")

    legendsize=7
    for idx, _ax in enumerate(axes):
        _ax.grid(False)
        _ax.set_title(None)
        _ax.set_ylabel(None)
        _ax.xaxis.set_tick_params(labelbottom=True)
        _ax.legend(loc=['best', 'upper left'][idx], fontsize=legendsize)

    axes[0].xaxis.set_tick_params(labelbottom=False)

    axes[0].set_ylabel(r"M $\rightarrow$ T"
                       "\n"
                       r"(nS)")
    axes[1].set_ylabel(r"T $\rightarrow$ T"
                       "\n"
                       r"(nS)")
    axes[1].set_xlabel('Trial')

    fig.tight_layout()
    fig.savefig(os.path.join(storage_paths['figures'], 'Fig_FF_weights_mean_{}_srcCol={}.pdf'.format(parameters.label, src_col_ff)))
    fig.savefig(os.path.join(storage_paths['figures'], 'Fig_FF_weights_mean_{}_srcCol={}.svg'.format(parameters.label, src_col_ff)), format='svg')
    logprint.info('Saved figure: Fig_FF_weights_mean_{}_srcCol={}.pdf'.format(parameters.label, src_col_ff))


def run(parameters, display=False, plot=True, save=True, load_inputs=False, src_col_ff=1):
    # ############################ SYSTEM
    # experiments parameters
    if not isinstance(parameters, ParameterSet):
        parameters = ParameterSet(parameters)

    storage_paths = set_storage_locations(parameters.kernel_pars.data_path, parameters.kernel_pars.data_prefix,
                                          parameters.label, save=save)

    logger.update_log_handles(job_name=parameters.label, path=storage_paths['logs'])
    plot_weights_mean(parameters, storage_paths, int(src_col_ff))


