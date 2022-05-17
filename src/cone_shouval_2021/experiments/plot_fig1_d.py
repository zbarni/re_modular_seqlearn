"""
Plots figure 1D.
"""
import itertools
import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import importlib.util

from fna.tools.visualization.helper import set_global_rcParams
from fna.tools.visualization.plotting import plot_raster, SpikePlots
from fna.tools.utils import logger
from fna.tools.utils.data_handling import set_storage_locations

from helper import plotting
from helper import fig_defaults as fig_def
from utils.parameters import ParameterSet

logprint = logger.get_logger(__name__)

def plot_rasters(parameters, storage_paths):
    """
    Plot some activity plots similar to Fig. 3.S4.b) from eLife version.
    """
    # assert 'T=3' in parameters.label, "Sorry, wrong trial! It's fixed to T=3 & test_batch=8"
    # if 'T=3' not in parameters.label:  #
    #     print("Currently we only want to plot T=3, test_batch_8")
    #     return

    plot_cols = [0, 1, 2, 3]
    colors = plotting.sns.color_palette('tab10')[:4]
    batch_to_plot = 8
    # T3, batch 8
    # T5, batch 8

    for batch_idx in [batch_to_plot]:
        filename = 'Recordings_{}.data'.format(parameters.label)
        fig, axes = plt.subplots(2, 1, figsize=fig_def.figsize['re_fig3s4_raster'])

        reward_times, recorded_data = plotting.load_recording_data(filename, storage_paths)
        batch_name = f'test_batch_{batch_idx}'
        pop_data = recorded_data['test_epoch'][batch_name]['pop_data']

        cvs_l5 = []
        isi_l5 = []
        cvs_l23 = []
        isi_l23 = []

        for col in plot_cols:
            for pop_name in ['E_L5_' + str(col), 'E_L23_' + str(col)]:
                pop_dict = pop_data[pop_name]
                spikelist = pop_dict['spikelists'][0]
                spikelist.id_offset(min(0, -200 * col) if 'L5' in pop_name else min(0, -200 * col - 100))
                spikelist.time_offset(-pop_dict['sampled_times'][0])

                times = spikelist.raw_data()[:, 0]
                neurons = spikelist.raw_data()[:, 1]
                # axes[0].plot(times, neurons, '.', color='k', ms=0.5, linewidth=0.5)
                axes[0].scatter(times, neurons, color='k', s=.2, linewidths=0.)
                isi = spikelist.isi()
                isi = np.array(list(itertools.chain(*[x.tolist() for x in isi])))
                isi = (isi[isi < 250]).tolist()
                if 'L5' in pop_name:
                    cvs_l5.extend(spikelist.cv_isi(float_only=True))
                    isi_l5.extend(isi)
                else:
                    cvs_l23.extend(spikelist.cv_isi(float_only=True))
                    isi_l23.extend(isi)

        for col in plot_cols:
            y = (0.2 * col / 0.8, (0.2 * col + 0.1) / 0.8)
            axes[0].axvspan(xmin=4900., xmax=5000., ymin=y[0], ymax=y[1], facecolor=colors[col], alpha=0.3)
            y = ((0.2 * col + 0.1) / 0.8, (0.2 * col + 0.2) / 0.8)
            axes[0].axvspan(xmin=4900., xmax=5000., ymin=y[0], ymax=y[1], facecolor=colors[col])

        hist1 = plotting.sns.histplot(isi_l5, ax=axes[1], color=plotting.sns.color_palette('tab10')[0], alpha=0.3,
                                      # label=f'Timer (CV = {np.round(np.mean(cvs_l5), decimals=2)})', bins=60,
                                      label=f'Timer', bins=60,
                                      element='step')
        ax_stats2 = plt.twinx()
        hist2 = plotting.sns.histplot(isi_l23, ax=ax_stats2, color=plotting.sns.color_palette('tab10')[0],
                                      # label=f'Messenger (CV = {np.round(np.mean(cvs_l23), decimals=2)})', bins=60,
                                      label=f'Messenger', bins=60,
                                      element='step')

        print(np.round(np.mean(cvs_l5), decimals=2))
        print(np.round(np.mean(cvs_l23), decimals=2))

        axes[0].axvspan(xmin=-150., xmax=-50., ymin=0., ymax=0.1 / 0.8, facecolor=colors[0], alpha=0.3)
        axes[0].set_ylim(1, 800)

        # added these three lines
        lines, labels = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax_stats2.get_legend_handles_labels()
        ax_stats2.legend(lines + lines2, labels + labels2, loc='best')

        axes[1].spines['right'].set_visible(True)
        ax_stats2.set_ylim(0, 150)

        for _ax in axes:
            _ax.grid(False)
            _ax.set_title(None)
            _ax.set_ylabel(None)
            _ax.margins(x=0.0)

        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        axes[0].yaxis.set_major_formatter(formatter)

        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        axes[1].yaxis.set_major_formatter(formatter)

        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax_stats2.yaxis.set_major_formatter(formatter)
        axes[1].minorticks_off()

        axes[0].set_ylabel('Neuron ID')
        axes[0].set_xlabel('Time (ms)')

        axes[1].set_ylabel('Count (Timer)')
        axes[1].set_xlabel('ISI (ms)')
        ax_stats2.set_ylabel('Count (Messenger)')

        fig.tight_layout()
        plt.subplots_adjust(left=0.13, right=0.84, bottom=0.12)
        fig.savefig(os.path.join(storage_paths['figures'], f'Fig_1D_rasters_{parameters.label}_{batch_name}.pdf'))
        fig.savefig(os.path.join(storage_paths['figures'], f'Fig_1D_rasters_{parameters.label}_{batch_name}.svg'), format='svg')


def run(parameters, display=False, plot=True, save=True, load_inputs=False):
    # ############################ SYSTEM
    # experiments parameters
    if not isinstance(parameters, ParameterSet):
        parameters = ParameterSet(parameters)

    storage_paths = set_storage_locations(parameters.kernel_pars.data_path, parameters.kernel_pars.data_prefix,
                                          parameters.label, save=save)

    logger.update_log_handles(job_name=parameters.label, path=storage_paths['logs'])


    plot_rasters(parameters, storage_paths)


