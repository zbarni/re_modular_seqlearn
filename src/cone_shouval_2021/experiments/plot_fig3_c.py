"""
Figure 3C.
Plots figure 5 S1 from original publication, robustness to variation in the learning parameters.
"""
import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
import importlib.util

from fna.tools.utils import logger
from fna.tools.utils.data_handling import set_storage_locations

from helper import plotting
from helper import fig_defaults as fig_def
from utils.parameters import ParameterSet

logprint = logger.get_logger(__name__)

def plot_multi_trial(parameters, storage_paths, ax, plot_traces=True):
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

    colors = plotting.sns.color_palette('tab10')[:4]
    # plotting.boxplot_multitrial(parameters, recorded_data, reward_times, ax, plot_rewards=False, colors=colors)
    plotting.barplot_multitrial(parameters, recorded_data, reward_times, ax, plot_rewards=False, colors=colors)

    test_batches = len(recorded_data['test_epoch'].keys())
    ax.set_title(f"Reported times, {test_batches} trials,\none learning instance")


def plot_multi_instance(parameters, storage_paths, ax, plot_traces=True):
    """
    Consider all trials of the dataset
    """
    main_param_file = os.path.join(storage_paths['main'], 'ParameterSpace.py')
    spec = importlib.util.spec_from_file_location("ParameterSpace", main_param_file)
    mod_param = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod_param)

    # trials = mod_param.ParameterRange['T']
    trials = list(range(1, 3))
    recorded_data = []
    reward_times = []
    for trial in trials:
        inst_label = parameters.label[:parameters.label.find('_T=')] + f'_T={trial}'
        filename = 'Recordings_{}.data'.format(inst_label)

        try:
            with open(os.path.join(storage_paths['activity'], filename), 'rb') as f:
                data = pickle.load(f)

            recorded_data.append(data['recorded_data'])
            reward_times.append(data['reward_times'])

        except Exception as e:
            logprint.error(f'Exception occured: {str(e)}.\n\tSkipping dataset {filename}')
            continue

    colors = plotting.sns.color_palette('tab10')[:4]
    plotting.barplot_multi_instance(parameters, recorded_data, ax, colors)


def run(parameters, display=False, plot=True, save=True, load_inputs=False, baseline_param_file=None,
        rand1_param_file=None):
    # ############################ SYSTEM
    # experiments parameters
    if not isinstance(parameters, ParameterSet):
        parameters = ParameterSet(parameters)

    storage_paths = set_storage_locations(parameters.kernel_pars.data_path, parameters.kernel_pars.data_prefix,
                                          parameters.label, save=save)

    logger.update_log_handles(job_name=parameters.label, path=storage_paths['logs'])

    panels = 2 if rand1_param_file is None else 3
    fig, axes = plt.subplots(nrows=1, ncols=panels, figsize=fig_def.figsize['re_fig5s1'], sharex=True)

    # plot randomized
    plot_multi_trial(parameters, storage_paths, axes[1])

    # load and plot baseline data using specified dataset
    logprint.info(f'Loading baseline dataset: {baseline_param_file}')
    parameters_baseline = ParameterSet(baseline_param_file)
    storage_paths_baseline = set_storage_locations(parameters_baseline.kernel_pars.data_path,
                                                   parameters_baseline.kernel_pars.data_prefix,
                                                   parameters_baseline.label, save=save)
    plot_multi_trial(parameters_baseline, storage_paths_baseline, axes[0])

    # load and plot baseline data using specified dataset
    logprint.info(f'Loading randomize_once dataset: {rand1_param_file}')
    parameters_rand1 = ParameterSet(rand1_param_file)
    storage_paths_rand1 = set_storage_locations(parameters_rand1.kernel_pars.data_path,
                                                   parameters_rand1.kernel_pars.data_prefix,
                                                   parameters_rand1.label, save=save)
    plot_multi_instance(parameters_rand1, storage_paths_rand1, axes[2])

    for _ax in axes:
        _ax.grid(False)
        _ax.set_title(None)
        _ax.set_ylabel(None)
        _ax.set_ylim(0., 1100.)
        _ax.set_yticks(np.arange(100, 1101, 200).astype(int))
        _ax.set_xticklabels([1, 2, 3, 4])
        _ax.set_xlabel('Column')
        _ax.yaxis.set_tick_params(labelbottom=True)

    axes[0].set_title('Non-randomized')
    axes[1].set_title('Trial-randomized')
    axes[2].set_title('Instance-randomized')
    axes[0].set_ylabel('Mean recall time')

    fig.tight_layout()
    cmp_trial = str(parameters_baseline.label[parameters_baseline.label.find('T='):])
    # fig.savefig(os.path.join(storage_paths['figures'], f'Fig5S1_barplot{parameters.label}_cmpBase-{cmp_trial}.pdf'))
    # fig.savefig(os.path.join(storage_paths['figures'], f'Fig5S1_barplot{parameters.label}_cmpBase-{cmp_trial}.svg'),
    #             format='svg')
    fig.savefig(os.path.join(storage_paths['figures'], f'Fig_3C_barplot{parameters.label}_cmpBase-{cmp_trial}.pdf'))
    fig.savefig(os.path.join(storage_paths['figures'], f'Fig_3C_barplot{parameters.label}_cmpBase-{cmp_trial}.svg'),
                format='svg')
