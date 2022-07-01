"""
Plots figure 2C, traces.
"""
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from fna.tools.utils import logger
from fna.tools.utils.data_handling import set_storage_locations

from helper import plotting
from helper import fig_defaults as fig_def
from utils.parameters import ParameterSet

logprint = logger.get_logger(__name__)


def plot_traces(parameters, storage_paths):
    """
    Plot Fig. 3.S2.b) from eLife version.
    """
    plot_cols = [0]  # plot FF connections from column C_0

    filename = 'Recordings_{}.data'.format(parameters.label)
    reward_times, recorded_data = plotting.load_recording_data(filename, storage_paths)
    data = plotting.get_traces(parameters, recorded_data, mean=True)

    # plot_trials = [3, 25, 35]  # which training trials to plot
    plot_trials = [1, 2, 3]  # which training trials to plot

    for col_id in plot_cols:
        fig, axes = plt.subplots(2, 3, figsize=fig_def.figsize['re_fig4s_traces'])

        # plot recurrent weights
        for ax_idx, trial in enumerate(plot_trials):
            # print(data['sampled_times'][col_id][0])
            trial_arr_idx = data['trial'][0].index(trial)
            axes[0][ax_idx].plot(np.arange(len(data['recurrent_ltp'][col_id][trial_arr_idx])),
                                 data['recurrent_ltp'][col_id][trial_arr_idx],
                                 '-', color='tomato', label='LTP')
            axes[0][ax_idx].plot(np.arange(len(data['recurrent_ltd'][col_id][trial_arr_idx])),
                                 data['recurrent_ltd'][col_id][trial_arr_idx],
                                 '--', color='cornflowerblue', label='LTD')

            axes[1][ax_idx].plot(np.arange(len(data['recurrent_ltp'][col_id][trial_arr_idx])),
                                 data['feedforward_ltp'][col_id][trial_arr_idx],
                                 '-', color='tomato', label='LTP')
            axes[1][ax_idx].plot(np.arange(len(data['recurrent_ltd'][col_id][trial_arr_idx])),
                                 data['feedforward_ltd'][col_id][trial_arr_idx],
                                 '--', color='cornflowerblue', label='LTD')

        sample_int = parameters.recording_pars.sampling_interval
        token_durations = parameters.input_pars.misc.token_duration
        tokens_sum = np.cumsum(token_durations)[:2]

        axes[0][0].get_shared_y_axes().join(axes[0][0], axes[0][1], axes[0][2])
        axes[1][0].get_shared_y_axes().join(*axes[1].tolist())

        for idx, _ax in enumerate(axes.flatten()):
            _ax.grid(False)
            _ax.set_title(None)
            _ax.set_ylabel(None)

            _ax.set_xticks([0] + tokens_sum / sample_int)
            _ax.set_xticklabels([0] + tokens_sum.astype(int))
            _ax.set_xlim([0, tokens_sum[-1] / sample_int + 200 / sample_int])
            if idx % 3 == 0:
                _ax.legend(loc='best')
            if idx > 2:
                _ax.set_xlabel('Time (ms)')

            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))
            _ax.yaxis.set_major_formatter(formatter)

        axes[0][0].set_ylabel('Recurrent\n(AU)')
        axes[1][0].set_ylabel('Feedforward\n(AU)')

        fig.tight_layout()
        fig.savefig(os.path.join(storage_paths['figures'], f'Fig_2C_traces_{parameters.label}_C={col_id}.pdf'))


def run(parameters, display=False, plot=True, save=True, load_inputs=False):
    # experiments parameters
    if not isinstance(parameters, ParameterSet):
        parameters = ParameterSet(parameters)

    storage_paths = set_storage_locations(parameters.kernel_pars.data_path, parameters.kernel_pars.data_prefix,
                                          parameters.label, save=save)
    logger.update_log_handles(job_name=parameters.label, path=storage_paths['logs'])

    # plot_weights_sum(parameters, storage_paths)
    plot_traces(parameters, storage_paths)


