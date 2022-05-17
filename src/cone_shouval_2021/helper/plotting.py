import logging
import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pickle

from . import input_seq
from . import naming
from . import fig_defaults as fig_def

try:
    matplotlib.rc_file('defaults/matplotlibrc')
except:
    logging.error("Could not find matplotlibrc!!! Figures may not be plotted as expected!")
    # raise

matplotlib.rcParams.update({
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

y_label_size = 18
plot_all_single_traces = False

def _get_col_id_from_name(name):
    return int(name.split('_')[-1])


def load_recording_data(filename, storage_paths):
    """
    Loads data recorded in the simulations.
    :param filename:
    :param storage_paths:
    :return:
    """
    try:
        with open(os.path.join(storage_paths['activity'], filename), 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f'Could not load recording data: {filename}!', str(e))
    return data['reward_times'], data['recorded_data']


def plot_summary(parameters, reward_times, recorded_data, storage_paths,
              plot_traces=True, savefig=False):
    """

    """
    rec_pars = parameters.recording_pars
    fs = 18
    binsize = 10.
    n_unique_trials = np.sum([len(b) for e in rec_pars.samples_to_record.values() for b in e.values()])

    #######################################################################
    # plot firing rates

    plot_epoch_batches = [('0', x) for x in recorded_data['0'].keys()]
    # plot_epoch_batches.insert(0, ('transient_epoch', 'transient_batch'))
    # plot_epoch_batches.append(('test_epoch', 'test_batch'))
    plot_epoch_batches.extend([('test_epoch', x) for x in recorded_data['test_epoch'].keys()])

    fig_dir = storage_paths['figures']
    fig, axes = plt.subplots(nrows=len(plot_epoch_batches), ncols=1, figsize=(10, len(plot_epoch_batches) * 3.))

    plot_psth(parameters, recorded_data, reward_times, plot_epoch_batches, axes, plot_labels=True, plot_inh_l5=True)

    # times = parameters.input_pars.misc.token_duration[:parameters.net_pars.n_cols]
    # times.insert(0, 0)

    for _ax in axes:
        _ax.grid(False)
        _ax.minorticks_on()
        scale_x = 1e3
        scale_y = 1e3
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale_x))
        _ax.xaxis.set_major_formatter(ticks_x)

        _ax.tick_params(axis='both', labelsize=fs)
        # _ax.legend(fontsize=10)
        _ax.xaxis.label.set_size(fs)
        _ax.xaxis.label.set_size(fs)

    fig.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, 'Fig3_{}_summary.pdf'.format(parameters.label)))


def plot_spiking_activity(parameters, reward_times, recorded_data, storage_paths,
              plot_traces=True, savefig=False):
    """
    Plot a figure containing the spiking activity for each trial
    """
    rec_pars = parameters.recording_pars
    fs = 18

    #######################################################################
    # plot firing rates

    plot_epoch_batches = [('0', x) for x in recorded_data['0'].keys()]
    plot_epoch_batches.append(('test_epoch', 'test_batch_0'))

    populations = parameters.net_pars.populations
    populations_to_plot = ['timer_exc']
    cols_to_plot = [0, 1]
    if len(populations) == 0:
        print("No additional population was found.")
        return

    fig_dir = storage_paths['figures']
    fig, axes = plt.subplots(nrows=len(plot_epoch_batches), ncols=len(cols_to_plot), figsize=(10, len(plot_epoch_batches) * 3))

    print("Plotting raster plot of {}...".format(populations))

    colors = ['blue', 'red', 'green', 'orange']

    ax_idx = 0
    for epoch, batch in plot_epoch_batches:
        print(recorded_data[epoch].keys())
        batch_data = recorded_data[epoch][batch]

        # get wanted sequence from batch
        sample_idx = parameters.recording_pars.samples_to_record[epoch][batch][0]
        seq_to_plot, seq_intervals = input_seq.get_sample_sequence_from_batch(batch_data, sample_idx)
        t_start_seq = seq_intervals[0][0]
        t_stop_seq = seq_intervals[-1][1]

        for col_id in cols_to_plot:
            pop_to_plot = [naming.get_pop_name(k, col_id) for k in populations_to_plot]
            for idx, pop_name in enumerate(pop_to_plot):
                if 'I' in pop_name:
                    continue
                pop_idx = populations.index(pop_name)
                label = '{}__{}'.format(batch_data['vocabulary'][col_id + 1], pop_name)  # +1 because EOS@0

                #######################################
                spikelist = batch_data['pop_data'][pop_name]['spikelists'][0]
                spikelist = spikelist.time_slice(t_start_seq, t_stop_seq)
                spikelist.raster_plot(ax=axes[ax_idx][col_id], display=False, color=colors[col_id], ms=1.)
                print("Mean rate", spikelist.mean_rate())
                print(f"# total spikes within interval: {np.sum(spikelist.spike_counts(dt=1.))}")

        # # plot reward bars
        # learning_pars = parameters.learning_pars
        # for t_ in reward_times[epoch][batch]:
        #     if t_ < t_start_seq or t_ > t_stop_seq:
        #         continue
        #     for ax in [axes[ax_idx]]:
        #         t_tmp = t_ - offset
        #         ax.axvspan(t_tmp, t_tmp + learning_pars.T_reward, facecolor='forestgreen', alpha=0.3)
        #         ax.axvspan(t_tmp + learning_pars.T_reward, t_tmp + learning_pars.T_reward + learning_pars.T_ref,
        #                    facecolor='silver', alpha=0.3)
        # axes[ax_idx].set_ylabel('neuron id', fontsize=y_label_size)
        ax_idx += 1

    for _ax in axes.flatten():
        _ax.grid(False)
        _ax.minorticks_on()
        # _ax.set_xlim(0., sim_time)
        # _ax.set_xticks(times)sxxsx
        scale_x = 1e3
        scale_y = 1e3
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale_x))
        _ax.xaxis.set_major_formatter(ticks_x)

        _ax.tick_params(axis='both', labelsize=fs)
        # _ax.legend(fontsize=10)
        _ax.xaxis.label.set_size(fs)
        _ax.xaxis.label.set_size(fs)

    fig.tight_layout()
    if savefig:
        # fig.savefig(os.path.join(fig_dir, 'Fig3_{}.pdf'.format(parameters.label)))
        fig.savefig(os.path.join(fig_dir, 'spikes_{}.pdf'.format(parameters.label)))


def plot_results(parameters, reward_times, recorded_data, storage_paths, fig_axes=None,
                 plot_traces=True, savefig=False):
    """

    """
    rec_pars = parameters.recording_pars
    fs = 18
    binsize = 10.
    n_unique_trials = np.sum([len(b) for e in rec_pars.samples_to_record.values() for b in e.values()])

    fig_dir = storage_paths['figures']

    if fig_axes is None:
        if plot_traces:
            n_rows = 5  # if len(snn.populations) == 1 else 8
            # fig, axes = plt.subplots(nrows=n_rows, ncols=n_unique_trials, figsize=(8 * n_unique_trials, 25))
            fig = plt.figure(figsize=(7 * n_unique_trials, n_rows * 3))
            axes = [[] for _ in range(n_rows)]

            for r in range(n_rows):
                ax_one = plt.subplot2grid((n_rows, n_unique_trials), (r, 0))
                axes[r].append(ax_one)
                axes[r].extend( [plt.subplot2grid((n_rows, n_unique_trials), (r, col), sharey=ax_one)
                                 for col in range(1, n_unique_trials)])

        else:
            n_rows = 2
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_unique_trials, figsize=(8 * n_unique_trials, 10))

    else:
        fig, axes = fig_axes

    # ax_raster = axes[0]
    # ax_input = axes[1]
    ax_filtered = axes[0]
    ax_psth = None #axes[1]

    # set title
    idx = 0
    abs_sample_idx = 0
    prev_batch_idx = 0
    try:
        for epoch in rec_pars.samples_to_record:
            for batch in rec_pars.samples_to_record[epoch]:
                if isinstance(batch, str) and 'test' in batch:
                    abs_sample_idx = batch
                else:
                    abs_sample_idx += (max(int(batch) - int(prev_batch_idx) - 1, 0)) * int(parameters.task_pars.batch_size)
                axes[0][idx].set_title('Epoch #{}, batch #{}, sample #{}'.format(epoch, batch, abs_sample_idx))
                abs_sample_idx += 1  # add current batch of size 1
                prev_batch_idx = batch
                idx += 1
    except:
        pass

    #######################################################################
    # plot firing rates
    plot_psth(parameters, recorded_data, reward_times, binsize, ax_filtered, ax_psth)

    if plot_traces:
        ax_mean_trace_T = axes[1]
        ax_mean_weight_T = axes[2]
        ax_mean_trace_M = axes[3]
        ax_mean_weight_M = axes[4]
        ax_mean_trace_diff_T = None # axes[6]

        plot_traces_and_weights(parameters, recorded_data, reward_times, ax_mean_trace_T, ax_mean_weight_T,
                                ax_mean_trace_M, ax_mean_weight_M, ax_mean_trace_diff_T)

    for _ax_batches in axes:
        for _ax in _ax_batches:
            _ax.grid(False)
            _ax.minorticks_on()
            # _ax.set_xlim(0., sim_time)
            # _ax.set_xticks(np.arange(0., sim_time, int(sim_time / 10)))
            scale_x = 1e3
            scale_y = 1e3
            ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale_x))
            _ax.xaxis.set_major_formatter(ticks_x)

            _ax.tick_params(axis='both', labelsize=fs)
            # _ax.legend(fontsize=10)
            _ax.xaxis.label.set_size(fs)
            _ax.xaxis.label.set_size(fs)

    fig.tight_layout()
    if savefig:
        fig.savefig(os.path.join(fig_dir, 'Summary_{}.pdf'.format(parameters.label)))


def plot_traces_and_weights(parameters, reward_times, recorded_data, storage_paths, savefig=False):
    """
    Plots a four column plot [Timer trace, Timer weight, FF trace, FF weight], with the number of rows dynamically
    adjustable depending how many batches we want to plot.
    :param parameters:
    :param reward_times:
    :param recorded_data:
    :param storage_paths:
    :param savefig:
    :return:
    """
    populations = parameters.net_pars.populations
    if len(populations) == 0:
        print("No additional population was found.")
        return

    print("Plotting eligibility traces and weights of {} population...".format(populations))

    colors = ['blue', 'red', 'green', 'orange']
    linestyle = ['-', '--']

    plot_epoch_batches = [('0', x) for x in recorded_data['0'].keys()]
    # plot_epoch_batches.insert(0, ('transient_epoch', 'transient_batch'))
    plot_epoch_batches.append(('test_epoch', 'test_batch'))

    fig_dir = storage_paths['figures']
    fig, axes = plt.subplots(nrows=len(plot_epoch_batches), ncols=4, figsize=(30, len(plot_epoch_batches) * 2))

    batch_idx = 0
    lw = 2.5
    for epoch, batch in plot_epoch_batches:
        batch_data = recorded_data[epoch][batch]

        # get wanted sequence from batch
        sample_idx = parameters.recording_pars.samples_to_record[epoch][batch][0]
        seq_to_plot, seq_intervals = input_seq.get_sample_sequence_from_batch(batch_data, sample_idx)
        t_start_seq = seq_intervals[0][0]
        t_stop_seq = seq_intervals[-1][1]

        for conn_name, conn_data in batch_data['conn_data'].items():
            # don't plot any feedback connections, so ensure source col <= target col
            if _get_col_id_from_name(conn_name[1]) > _get_col_id_from_name(conn_name[0]):
                continue

            times = np.array(conn_data['sampled_times'])
            valid_indices = np.where((times >= t_start_seq) & (times <= t_stop_seq))[0]

            times = times[valid_indices]
            ltp = np.array(conn_data['sampled_ltp'])[valid_indices]
            ltd = np.array(conn_data['sampled_ltd'])[valid_indices]
            weights = np.array(conn_data['sampled_weights'])[valid_indices]

            # timer connections
            if conn_name[0] == conn_name[1]:
                col_id = _get_col_id_from_name(conn_name[1])
                c = colors[col_id]
                axes[batch_idx][0].plot(times, ltp, linestyle[0], color=c, label='LTP {}'.format(col_id), lw=lw)
                axes[batch_idx][0].plot(times, ltd, linestyle[1], color=c, label='LTD {}'.format(col_id), lw=lw)
                axes[batch_idx][1].plot(times, weights, '-', color=c, label='weight {}'.format(col_id), lw=lw)
                # ax_mean_trace_diff_T[ax_idx].plot(times, ltp-ltd, '-', color=c, label='LTP-LTD {}'.format(col_id))

            # feed-forward connections
            else:
                src_col_id = _get_col_id_from_name(conn_name[1])  # get source column
                c = colors[src_col_id]
                axes[batch_idx][2].plot(times, ltp, linestyle[0], color=c, label='FF LTP {}'.format(src_col_id), lw=lw)
                axes[batch_idx][2].plot(times, ltd, linestyle[1], color=c, label='FF LTD {}'.format(src_col_id), lw=lw)
                axes[batch_idx][3].plot(times, weights, '-', color=c, label='FF weight {}'.format(src_col_id), lw=lw)

        # plot reward bars
        learning_pars = parameters.learning_pars
        for t_ in reward_times[epoch][batch]:
            if t_ < t_start_seq or t_ > t_stop_seq:
                continue
            for plot_col in range(0, len(axes[0])):
                axes[batch_idx][plot_col].axvspan(t_, t_ + learning_pars.T_reward, facecolor='forestgreen', alpha=0.3)
                axes[batch_idx][plot_col].axvspan(t_ + learning_pars.T_reward, t_ + learning_pars.T_reward
                                                  +learning_pars.T_ref, facecolor='silver', alpha=0.3)


    # ax.fill_between(times, mean_ltp - std_ltp, mean_ltp - std_ltp, facecolors='g', alpha=0.3)
    # ax.fill_between(times, mean_ltd - std_ltd, mean_ltd - std_ltd, facecolors='r', alpha=0.3)

    # n_syn_rec_mean = parameters.recording_pars.n_syn_rec_mean
        for plot_col in range(0, len(axes[0])):
            axes[batch_idx][plot_col].set_ylabel('AU', fontsize=y_label_size)
            # ax_mean_trace_T[batch_idx].set_title('Mean traces computed over {} synapses'.format(n_syn_rec_mean))
            axes[batch_idx][plot_col].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))
            axes[batch_idx][plot_col].set_ylabel('nS', fontsize=y_label_size)
            # ax_mean_weight_T[batch_idx].set_title('Mean weights computed over {} synapses'.format(n_syn_rec_mean))
            axes[batch_idx][plot_col].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))

        batch_idx += 1

    axes[0][0].set_title("Mean T traces")
    axes[0][1].set_title("Mean T weights")
    axes[0][2].set_title("Mean FF traces")
    axes[0][3].set_title("Mean FF weights")
    if savefig:
        fig.savefig(os.path.join(fig_dir, 'Fig_traces_{}.pdf'.format(parameters.label)))


def plot_weights_per_trial(parameters, reward_times, recorded_data, storage_paths, savefig=False):
    """
    Plots the recurrent and FF weights as a function of trials (batches).
    :param parameters:
    :param reward_times:
    :param recorded_data:
    :param storage_paths:
    :param savefig:
    :return:
    """
    populations = parameters.net_pars.populations
    if len(populations) == 0:
        print("No additional population was found.")
        return

    print("Plotting eligibility traces and weights of {} population...".format(populations))

    # colors = ['blue', 'red', 'green', 'orange']
    colors = sns.color_palette('tab10')[:6]
    linestyle = ['-', '--']

    plot_epoch_batches = [('0', x) for x in recorded_data['0'].keys()]
    # plot_epoch_batches.insert(0, ('transient_epoch', 'transient_batch'))
    # plot_epoch_batches.append(('test_epoch', 'test_batch'))

    fig_dir = storage_paths['figures']
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    batch_idx = 0
    lw = 2.5
    res_weights = None
    res_batches = []
    for epoch, batch in plot_epoch_batches:
        batch_data = recorded_data[epoch][batch]
        if res_weights is None:  # init results dictionary
            res_weights = {cn: [] for cn in batch_data['conn_data']}
        res_batches.append(int(batch))

        # get wanted sequence from batch
        sample_idx = parameters.recording_pars.samples_to_record[epoch][batch][0]
        seq_to_plot, seq_intervals = input_seq.get_sample_sequence_from_batch(batch_data, sample_idx)
        t_start_seq = seq_intervals[0][0]
        t_stop_seq = seq_intervals[-1][1]

        for conn_name, conn_data in batch_data['conn_data'].items():
            # don't plot any feedback connections, so ensure source col <= target col
            if _get_col_id_from_name(conn_name[1]) > _get_col_id_from_name(conn_name[0]):
                continue

            times = np.array(conn_data['sampled_times'])
            valid_indices = np.where((times >= t_start_seq) & (times <= t_stop_seq))[0]

            times = times[valid_indices]
            weights = np.array(conn_data['sampled_weights'])[valid_indices]
            if len(weights):
                res_weights[conn_name].append(weights[-1])
            else:
                res_weights[conn_name].append(np.nan)

    res_batches = np.array(res_batches) + 1
    for conn_name in res_weights.keys():
        # timer connections
        if conn_name[0] == conn_name[1]:
            col_id = _get_col_id_from_name(conn_name[1])
            c = colors[col_id]
            axes[1].plot(res_batches, res_weights[conn_name], '-', color=c, label='C{}'.format(col_id), lw=lw)

        # feed-forward connections
        else:
            src_col_id = _get_col_id_from_name(conn_name[1])  # get source column
            if len(res_weights[conn_name]):
                c = colors[src_col_id]
                axes[0].plot(res_batches, res_weights[conn_name], '-', color=c,
                             label='C{} -> C{}'.format(src_col_id, src_col_id + 1), lw=lw)
            else:
                print("WARNING: Summary plot doesn't currently support all-to-all FF connections! SKipping.")

    axes[0].set_ylabel('Mean FF weight', fontsize=y_label_size)
    axes[1].set_ylabel('Mean rec weight', fontsize=y_label_size)

    for plot_col in range(2):
        axes[plot_col].legend(loc='best')
        axes[plot_col].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))
        axes[plot_col].set_xlabel('trial')
    if savefig:
        fig.savefig(os.path.join(fig_dir, 'Fig_weights_{}.pdf'.format(parameters.label)))


def plot_single_psth(parameters, recorded_data, reward_times, epoch, batch, ax, plot_rewards=True,
                     colors=None, plot_labels=False, plot_inh_l5=True, linewidths=None, plot_inh_l23=None):
    """
    Plots the PSTH for a single epoch/batch of T, I(T) and M populations in each column.

    :return:
    """
    populations = parameters.net_pars.populations
    populations_to_plot = ['timer_exc', 'messenger_exc']
    if plot_inh_l5:
        populations_to_plot += ['timer_inh']
    if plot_inh_l23:
        populations_to_plot += ['messenger_inh']

    if len(populations) == 0:
        print("No additional population was found.")
        return

    print("Plotting PSTH of {}...".format(populations))

    colors = sns.color_palette('tab10')[:4] if colors is None else colors
    linewidths = [1.5, 1.5, 1.5, 1.5] if linewidths is None else linewidths
    linestyle = ['-', '-', '--', ':']
    marker = [None, None,None,None]
    alpha = [0.3, 1., 0.3, 1.]

    batch_data = recorded_data[epoch][batch]

    # get wanted sequence from batch
    sample_idx = parameters.recording_pars.samples_to_record[epoch][batch][0]
    seq_to_plot, seq_intervals = input_seq.get_sample_sequence_from_batch(batch_data, sample_idx)
    t_start_seq = seq_intervals[0][0]
    t_stop_seq = seq_intervals[-1][1]
    offset = None # time offset from 0 for the current batch, to plot everything relative to 0

    for col_id in range(parameters.net_pars.n_cols):
        pop_to_plot = [naming.get_pop_name(k, col_id) for k in populations_to_plot]
        for idx, pop_name in enumerate(pop_to_plot):
            if plot_labels is False:
                label = None
            elif plot_labels is None:
                label = '{}__{}'.format(batch_data['vocabulary'][col_id + 1], pop_name)  # +1 because EOS@0
            else:
                label = plot_labels[col_id][idx]

            times = np.array(batch_data['pop_data'][pop_name]['sampled_times'])
            rates = np.array(batch_data['pop_data'][pop_name]['sampled_rates'])
            offset = min(times) if offset is None else offset

            if linestyle[idx] == '--':
                dashes = (3, 2)
            elif linestyle[idx] == ':':
                dashes = (1, 2)
            else:
                dashes = (None, None)

            if idx < 2:
                ax.plot(times - offset, rates * 1000.,  #/ parameters.learning_pars.tau_w,
                        alpha=alpha[idx], color=colors[col_id], label=label, marker=marker[idx],
                        linewidth=linewidths[idx], linestyle=linestyle[idx], dashes=dashes)
            else:
                ith = 10
                ax.plot((times - offset)[::ith], rates[::ith] * 1000.,  # / parameters.learning_pars.tau_w,
                        alpha=alpha[idx], color=colors[col_id], label=label, marker=marker[idx], markersize=1.,
                        linewidth=linewidths[idx], linestyle=linestyle[idx], dashes=dashes)

    # plot reward bars
    if plot_rewards:
        learning_pars = parameters.learning_pars
        for t_ in reward_times[epoch][batch]:
            if t_ < t_start_seq or t_ > t_stop_seq:
                continue

            t_tmp = t_ - offset
            ax.axvspan(t_tmp, t_tmp + learning_pars.T_reward, facecolor='forestgreen', alpha=0.3)
            ax.axvspan(t_tmp + learning_pars.T_reward, t_tmp + learning_pars.T_reward + learning_pars.T_ref,
                       facecolor='silver', alpha=0.3)


def plot_psth(parameters, recorded_data, reward_times, plot_epoch_batches, axes_filtered, plot_rewards=True,
              colors=None, plot_labels=False, plot_inh_l5=False):
    """
    Plots the PSTH of T, I(T) and M populations in each column, for the last trial within
    each recorded batch and epoch.

    :return:
    """
    populations = parameters.net_pars.populations
    populations_to_plot = ['timer_exc', 'messenger_exc']
    if plot_inh_l5:
        populations_to_plot += ['timer_inh']
        populations_to_plot += ['messenger_inh']

    if len(populations) == 0:
        print("No additional population was found.")
        return

    print("Plotting PSTH of {}...".format(populations))

    # colors = ['teal', 'indigo', 'maroon', 'black']
    # colors = ['blue', 'red', 'green', 'orange']
    colors = sns.color_palette('tab10')[:6] if colors is None else colors
    linewidths = [1.5, 1.5, 1.5, 1.5]
    linestyle = ['-', '-', '--', ':']
    alpha = [0.3, 1., 0.3, 1.]

    ax_idx = 0
    for epoch, batch in plot_epoch_batches:
        batch_data = recorded_data[epoch][batch]

        # get wanted sequence from batch
        sample_idx = parameters.recording_pars.samples_to_record[epoch][batch][0]
        seq_to_plot, seq_intervals = input_seq.get_sample_sequence_from_batch(batch_data, sample_idx)
        t_start_seq = seq_intervals[0][0]
        t_stop_seq = seq_intervals[-1][1]
        offset = None # time offset from 0 for the current batch, to plot everything relative to 0

        for col_id in range(parameters.net_pars.n_cols):
            pop_to_plot = [naming.get_pop_name(k, col_id) for k in populations_to_plot]
            for idx, pop_name in enumerate(pop_to_plot):
                # label = '{}__{}'.format(batch_data['vocabulary'][col_id + 1], pop_name)  # +1 because EOS@0
                label = '{}'.format(pop_name)  # +1 because EOS@0

                times = np.array(batch_data['pop_data'][pop_name]['sampled_times'])
                rates = np.array(batch_data['pop_data'][pop_name]['sampled_rates'])
                offset = min(times) if offset is None else offset

                axes_filtered[ax_idx].plot(times - offset, rates * 1000., #/ parameters.learning_pars.tau_w,
                                           alpha=alpha[idx], color=colors[col_id], label=label,
                                           linewidth=linewidths[idx], linestyle=linestyle[idx])
                # print(f"Mean rates (PSTH): {np.mean((rates * 1000))}")

        # plot reward bars
        if plot_rewards:
            if batch != 'transient_batch' and 'test' not in batch:
                learning_pars = parameters.learning_pars
                for t_ in reward_times[epoch][batch]:
                    if t_ < t_start_seq or t_ > t_stop_seq:
                        continue
                    for ax in [axes_filtered[ax_idx]]:
                        t_tmp = t_ - offset
                        ax.axvspan(t_tmp, t_tmp + learning_pars.T_reward, facecolor='forestgreen', alpha=0.3)
                        ax.axvspan(t_tmp + learning_pars.T_reward, t_tmp + learning_pars.T_reward + learning_pars.T_ref,
                                   facecolor='silver', alpha=0.3)

        if plot_labels:
            axes_filtered[ax_idx].set_title(f'Training trial #{int(batch)}' if epoch == '0'
                                            else f'Replay trial #{batch}')
        # axes_filtered[ax_idx].set_ylabel('firing rates', fontsize=12)
        ax_idx += 1


def __get_reported_times(parameters, recorded_data, epoch, populations_to_plot):
    """
    For a given dataset/experiment/simulation instance, it returns the reported replay times for each sequence.
    The replay time is taken as the time from stimulus onset until the rate drops below a specific threshold (10 Hz).

    :param parameters:
    :param recorded_data:
    :param epoch:
    :param populations_to_plot:
    :return: [dict]
    """
    offset_th = 0.01
    reported_times = {}

    for batch in recorded_data[epoch]:
        batch_data = recorded_data[epoch][batch]

        for col_id in range(parameters.net_pars.n_cols):
            pop_to_plot = [naming.get_pop_name(k, col_id) for k in populations_to_plot]
            for idx, pop_name in enumerate(pop_to_plot):
                if pop_name not in reported_times:  # insert new pop
                    reported_times[pop_name] = []

                # fixed static onset time relative to 0
                # onset_times_static = np.cumsum([0.] + parameters.input_pars.misc.token_duration[:-1])

                times = np.array(batch_data['pop_data'][pop_name]['sampled_times'])
                rates = np.array(batch_data['pop_data'][pop_name]['sampled_rates'])
                # rep_time = times[len(times) - np.argmax(rates[::-1] > rate_th)] - min(times)

                # empirically chosen rate threshold to signal active column
                # onset_th = 0.05 if col_id == 0 else 0.08
                onset_th = np.max(rates)
                # slice arrays from stimulus onset / active onwards
                times_after_onset = times[np.argmax(rates >= onset_th):]
                rates_after_onset = rates[np.argmax(rates >= onset_th):]
                rep_time = times_after_onset[np.argmin(rates_after_onset > offset_th)] - min(times)

                # ignore reported times under 200 ms, probably some weird activity going on..
                if rep_time > 200.:
                    reported_times[pop_name].append(rep_time)

    return reported_times


def boxplot_multitrial(parameters, recorded_data, reward_times, ax, plot_rewards=True, colors=None):
    """

    :return:
    """
    populations = parameters.net_pars.populations
    populations_to_plot = ['timer_exc']
    if len(populations) == 0:
        print("No additional population was found.")
        return

    print("Plotting multitrial boxplot of {}...".format(populations))

    colors = sns.color_palette('tab10')[:6] if colors is None else colors

    epoch = 'test_epoch'
    reported_times = __get_reported_times(parameters, recorded_data, epoch, populations_to_plot)

    flierprops = dict(markerfacecolor='tab:red', markersize=4, linestyle='none', markeredgecolor='k',
                      markeredgewidth=0.5)
    box_plot = sns.boxplot(data=list(reported_times.values()), ax=ax, palette=colors, width=0.7, linewidth=1.,
                           showfliers=True, flierprops=flierprops)
    medians = [np.median(np.array(x)) for x in list(reported_times.values())]

    for xtick in box_plot.get_xticks():
        # box_plot.text(xtick + 0.65, medians[xtick] - 20., medians[xtick],
        #               horizontalalignment='center', size='small', color='k', weight='semibold')
        logging.info(f"Median for tick {xtick} is {medians[xtick]} (multitrial)")

    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .5))


def barplot_multitrial(parameters, recorded_data, reward_times, ax, plot_rewards=True, colors=None):
    """

    :return:
    """
    populations = parameters.net_pars.populations
    populations_to_plot = ['timer_exc']
    if len(populations) == 0:
        print("No additional population was found.")
        return

    print("Plotting multitrial barplot of {}...".format(populations))

    colors = sns.color_palette('tab10')[:6] if colors is None else colors
    colors = [sns.desaturate(x, 0.75) for x in colors]

    epoch = 'test_epoch'
    reported_times = __get_reported_times(parameters, recorded_data, epoch, populations_to_plot)

    onset_times = np.cumsum([0] + parameters.input_pars.misc.token_duration[:-1])
    reported_times = [np.array(x) - onset_times[idx] for idx, x in enumerate(reported_times.values())]

    mean = [np.mean(x) for x in reported_times]
    std = [np.std(x) for x in reported_times]
    # bar_plot = sns.barplot(data=list(reported_times), ax=ax, palette=colors, linewidth=0.3, ci='sd', capsize=0.1,
    #                        saturation=1.)
    bar_plot = ax.bar(np.arange(1, parameters.net_pars.n_cols + 1), mean, yerr=std, color=colors,
                      linewidth=0.6, width=0.6)
    ax.set_xticks(np.arange(1, parameters.net_pars.n_cols + 1))


def barplot_multi_instance(parameters, recorded_data_list, ax, colors=None):
    """

    :return:
    """
    populations = parameters.net_pars.populations
    populations_to_plot = ['timer_exc']
    if len(populations) == 0:
        print("No additional population was found.")
        return

    print("Plotting multitrial barplot of {}...".format(populations))

    colors = sns.color_palette('tab10')[:6] if colors is None else colors
    colors = [sns.desaturate(x, 0.75) for x in colors]

    epoch = 'test_epoch'

    reported_times_concatenated = {}

    for rec_data in recorded_data_list:
        reported_times_abs = __get_reported_times(parameters, rec_data, epoch, populations_to_plot)

        # align to onset
        onset_times = np.cumsum([0] + parameters.input_pars.misc.token_duration[:-1])
        reported_times_aligned = [np.array(x) - onset_times[idx] for idx, x in enumerate(reported_times_abs.values())]

        for col_id, rt in enumerate(reported_times_aligned):
            if col_id not in reported_times_concatenated:
                reported_times_concatenated[col_id] = []

            reported_times_concatenated[col_id].extend(rt)

    bar_plot = ax.bar(np.arange(1, parameters.net_pars.n_cols + 1),
                      [np.mean(col_res) for col_res in reported_times_concatenated.values()],
                      yerr=[np.std(col_res) for col_res in reported_times_concatenated.values()],
                      color=colors,
                      linewidth=0.6, width=0.6)
    ax.set_xticks(np.arange(1, parameters.net_pars.n_cols + 1))


def boxplot_multi_instance(parameters, recorded_data_list, ax):
    """

    :return:
    """
    populations = parameters.net_pars.populations
    populations_to_plot = ['timer_exc']
    if len(populations) == 0:
        print("No additional population was found.")
        return

    print("Plotting multi-instance boxplot of {}...".format(populations))

    colors = sns.color_palette('tab10')[:6]

    epoch = 'test_epoch'
    reported_median_times = {}

    for rec_data in recorded_data_list:
        reported_times = __get_reported_times(parameters, rec_data, epoch, populations_to_plot)

        for col_id, rt in reported_times.items():
            if col_id not in reported_median_times:
                reported_median_times[col_id] = []

            reported_median_times[col_id].append(np.median(rt))

    flierprops = dict(markerfacecolor='tab:red', markersize=4, linestyle='none', markeredgecolor='k',
                      markeredgewidth=0.5)
    box_plot = sns.boxplot(data=list(reported_median_times.values()), ax=ax, palette=colors, width=0.7, linewidth=1.,
                           showfliers=True, flierprops=flierprops)
    medians = [np.median(np.array(x)) for x in list(reported_median_times.values())]

    for xtick in box_plot.get_xticks():
        # box_plot.text(xtick + 0.95, medians[xtick] - 20., medians[xtick],
        #               horizontalalignment='center', size=7, color='k', weight='semibold')
        logging.info(f"Median for tick {xtick} is {medians[xtick]} (multiinstance)")

    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .5))


def plot_single_traces(traces, c_name, batch, parameters, storage_paths):
    """
    Plots individual eligibility traces for the connection `c_name`.
    """
    if 'plot_single_traces' in parameters.plotting_pars and \
        not isinstance(parameters.plotting_pars.plot_single_traces, bool):

        fig, axes = plt.subplots(nrows=2, ncols=1)
        # n_colors = float(max(5, plot_cnt))
        axes[0].set_prop_cycle(color=[plt.get_cmap('jet')(i) for i in np.linspace(0, 1, 30)])
        axes[1].set_prop_cycle(color=[plt.get_cmap('jet')(i) for i in np.linspace(0, 1, 30)])

        for c in traces:
            axes[0].plot(c[0], c[2], '-', linewidth=.5)
            axes[1].plot(c[0], c[3], '--', linewidth=.5)

        fig.savefig(os.path.join(storage_paths['figures'], '{}_#batch{}_{}.pdf'.format(c_name, batch,
                                                                                       parameters.label)))


def get_weights(parameters, recorded_data, mean=False):
    """
    """
    populations = parameters.net_pars.populations
    n_cols = parameters.net_pars.n_cols
    if len(populations) == 0:
        print("No additional population was found.")
        return

    plot_epoch_batches = [('0', x) for x in recorded_data['0'].keys()]
    # plot_epoch_batches.append(('test_epoch', 'test_batch'))

    results = {
        'trial': [[] for _ in range(n_cols)],
        'recurrent': [[] for _ in range(n_cols)],
        'feedforward': [[] for _ in range(n_cols - 1)]
    }
    for epoch, batch in plot_epoch_batches:
        batch_data = recorded_data[epoch][batch]

        # get wanted sequence from batch
        sample_idx = parameters.recording_pars.samples_to_record[epoch][batch][0]
        seq_to_plot, seq_intervals = input_seq.get_sample_sequence_from_batch(batch_data, sample_idx)
        t_start_seq = seq_intervals[0][0]
        t_stop_seq = seq_intervals[-1][1]

        for conn_name, conn_data in batch_data['conn_data'].items():
            # don't plot any feedback connections, so ensure source col <= target col
            if _get_col_id_from_name(conn_name[1]) > _get_col_id_from_name(conn_name[0]):
                continue

            times = np.array(conn_data['sampled_times'])
            valid_indices = np.where((times >= t_start_seq) & (times <= t_stop_seq))[0]

            weight_key = 'sampled_weights' if mean else 'sampled_weights_sum'
            weights = np.array(conn_data[weight_key])[valid_indices]
            col_id = _get_col_id_from_name(conn_name[1])

            if conn_name[0] == conn_name[1]:  # timer connections
                results['recurrent'][col_id].append(weights[-1])  # append last sampled weight to array
                results['trial'][col_id].append(int(batch) + 1)  # append trial
            else:  # feed-forward connections
                results['feedforward'][col_id].append(weights[-1])  # append last sampled weight to array

    return results


def get_weights_ff_all(parameters, recorded_data, mean=False, source_col=0):
    """
    """
    populations = parameters.net_pars.populations
    n_cols = parameters.net_pars.n_cols
    if len(populations) == 0:
        print("No additional population was found.")
        return

    plot_epoch_batches = [('0', x) for x in recorded_data['0'].keys()]

    results = {
        'trial': [[] for _ in range(n_cols)],
        'recurrent': [[] for _ in range(n_cols)],
        'feedforward': [[] for _ in range(n_cols)]
    }
    for epoch, batch in plot_epoch_batches:
        batch_data = recorded_data[epoch][batch]

        # get wanted sequence from batch
        sample_idx = parameters.recording_pars.samples_to_record[epoch][batch][0]
        seq_to_plot, seq_intervals = input_seq.get_sample_sequence_from_batch(batch_data, sample_idx)
        t_start_seq = seq_intervals[0][0]
        t_stop_seq = seq_intervals[-1][1]

        for conn_name, conn_data in batch_data['conn_data'].items():
            times = np.array(conn_data['sampled_times'])
            valid_indices = np.where((times >= t_start_seq) & (times <= t_stop_seq))[0]

            weight_key = 'sampled_weights' if mean else 'sampled_weights_sum'
            weights = np.array(conn_data[weight_key])[valid_indices]
            src_col_id = _get_col_id_from_name(conn_name[1])

            if conn_name[0] == conn_name[1]:  # timer connections
                results['recurrent'][src_col_id].append(weights[-1])  # append last sampled weight to array
                results['trial'][src_col_id].append(int(batch) + 1)  # append trial

            # feed-forward connections
            elif src_col_id == source_col:
                tgt_col_id = _get_col_id_from_name(conn_name[0])
                results['feedforward'][tgt_col_id].append(weights[-1])  # append last sampled weight to array

    return results


def get_traces(parameters, recorded_data, mean=False):
    """
    """
    populations = parameters.net_pars.populations
    n_cols = parameters.net_pars.n_cols
    if len(populations) == 0:
        print("No additional population was found.")
        return

    plot_epoch_batches = [('0', x) for x in recorded_data['0'].keys()]
    # plot_epoch_batches.append(('test_epoch', 'test_batch'))

    results = {
        'trial': [[] for _ in range(n_cols)],
        'recurrent_ltp': [[] for _ in range(n_cols)],
        'recurrent_ltd': [[] for _ in range(n_cols)],
        'feedforward_ltp': [[] for _ in range(n_cols - 1)],
        'feedforward_ltd': [[] for _ in range(n_cols - 1)],
        'sampled_times': [[] for _ in range(n_cols)]
    }

    for epoch, batch in plot_epoch_batches:
        batch_data = recorded_data[epoch][batch]

        # get wanted sequence from batch
        sample_idx = parameters.recording_pars.samples_to_record[epoch][batch][0]
        seq_to_plot, seq_intervals = input_seq.get_sample_sequence_from_batch(batch_data, sample_idx)
        t_start_seq = seq_intervals[0][0]
        t_stop_seq = seq_intervals[-1][1]

        for conn_name, conn_data in batch_data['conn_data'].items():
            # don't plot any feedback connections, so ensure source col <= target col
            if _get_col_id_from_name(conn_name[1]) > _get_col_id_from_name(conn_name[0]):
                continue

            times = np.array(conn_data['sampled_times'])
            valid_indices = np.where((times >= t_start_seq) & (times <= t_stop_seq))[0]

            ltp = np.array(conn_data['sampled_ltp'])[valid_indices]
            ltd = np.array(conn_data['sampled_ltd'])[valid_indices]
            col_id = _get_col_id_from_name(conn_name[1])

            if conn_name[0] == conn_name[1]:  # timer connections
                results['recurrent_ltp'][col_id].append(ltp)
                results['recurrent_ltd'][col_id].append(ltd)
                results['sampled_times'][col_id].append(times)
                results['trial'][col_id].append(int(batch) + 1)  # append trial
            else:  # feed-forward connections
                results['feedforward_ltp'][col_id].append(ltp)
                results['feedforward_ltd'][col_id].append(ltd)

    return results


class OOMFormatter(ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)

    def _set_orderOfMagnitude(self, nothing=None):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % ticker._mathdefault(self.format)


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    assert len(points.shape) == 1
    median = np.median(points)
    diff = np.abs(points - median)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh