import logging
import time
import os
import pickle as pkl
import numpy as np
import itertools
from matplotlib import pyplot as plt
from multiprocessing.dummy import Pool
import threading
import nest

from helper import naming
from helper import input_seq
from helper import plotting
from . import is_etrace
from fna.tools.utils.operations import copy_dict
from fna.tools.utils import logger

logprint = logger.get_logger(__name__)


def _validate_traces_and_weights(data_, status_conns, times, batch_data, reward_times_dict):
    """
    Ensure that weights are changed only during reward period!!!
    :param status_conns:
    :param times:
    :param batch_data:
    :return:
    """
    # list of all current and previous reward times, for cross-batch consistency
    all_reward_times = sorted(itertools.chain(*[x for v_epoch in reward_times_dict.values() for x in v_epoch.values()]))

    def __is_reward_window(t_):
        for ep in all_reward_times:
            if ep <= t_ <= ep + data_.T_reward:
                return True
        return False

    error = False

    for c_idx, c in enumerate(status_conns):
        weights = c[4]
        times = c[0]
        for idx, t in enumerate(times):
            if idx == 0:
                continue

            # if not in reward window, weights should be unchanged!
            if not __is_reward_window(t) and not __is_reward_window(times[idx - 1]) \
                    and not np.isclose(weights[idx], weights[idx - 1], atol=1e-4):
                print('t: {}, t-1: {} w: {} vs {}'.format(t, times[idx - 1], weights[idx - 1], weights[idx]))
                error = True
                # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15,10))
                # axes[0].plot(times[idx-5:idx+5], c[2][idx-5:idx+5], marker='o', linestyle='None')
                # axes[1].plot(times[idx-5:idx+5], weights[idx-5:idx+5], marker='o', linestyle='None')
                #
                # for ax in axes:
                #     for t_rew in all_reward_times:
                #         ax.axvspan(t_rew, t_rew + data_.T_reward, facecolor='forestgreen', alpha=0.3)
                #         ax.axvspan(t_rew + data_.T_reward, t_rew + data_.T_reward + data_.T_ref, facecolor='silver',
                #                    alpha=0.3)
                #         if t_rew > max(times[idx-5:idx+5]):
                #             break
                #
                # fig.savefig('WTFFF.pdf')
    return not error


def _get_recurrent_weights(data):
    """
    Computes a (sparse) weight matrix according to a Gaussian/uniform distribution
    :param data:
    :return:
    """
    W = None
    if hasattr(data, 'w_gaussian') and data.w_gaussian:
        std_frac = 5. if not hasattr(data, 'w_TT_rec_learned_std') else data.w_TT_rec_learned_std
        print("Applying Gaussian randomization to the recurrent Timer weights with std_frac: {}".format(std_frac))
        std = data.w_TT_rec_learned / std_frac
        W = np.random.normal(data.w_TT_rec_learned, std, (data.N, data.N))
        W = np.clip(W, 0., 1000.)

    if hasattr(data, 'w_dist_uniform') and data.w_dist_uniform > 0:
        offset = data.w_TT_rec_learned / data.w_dist_uniform
        print("Applying uniform randomization to the recurrent Timer weights within interval [{}, {}]".format(
            data.w_TT_rec_learned - offset, data.w_TT_rec_learned + offset))

        W = np.random.uniform(data.w_TT_rec_learned - offset, data.w_TT_rec_learned + offset,
                              (data.N, data.N))
        for i in range(data.N):
            W[i][i] = 0.  # remove autapses for sure

    # enforce sparseness if needed by removing connections
    if hasattr(data, 'sparse') and data.sparse and W is not None:
        print('Creating a sparse recurrent T->T connectivity...')
        # create a mask of indices to remove
        n_del = int((1 - data.epsilon) * (data.N * data.N))
        remove_mask = np.random.choice(np.arange(0, data.N * data.N).astype(int), n_del, replace=False)

        # flatten W set masked indices to 0
        flattened_W = W.flatten()
        flattened_W[remove_mask] = 0

        # create 2D matrix again
        W = flattened_W.reshape((data.N, data.N))
        print('Nonzero entries in spares T->T matrix: {}'.format(np.count_nonzero(W)))

    return W


def _get_plastic_connections(snn):
    """
    Returns the plastic connections, currently T->T and inter-columnar M->T
    :param snn:
    :return: [dict] two keys for recurrent and ff synapses, containing all such connections
    """
    plastic_conns = {}
    timer = []
    messenger = []
    n_cols = snn.n_populations // 4

    for col_id in range(n_cols):
        timer.extend(snn.find_population(naming.get_pop_name('timer_exc', col_id)).gids)
        messenger.extend(snn.find_population(naming.get_pop_name('messenger_exc', col_id)).gids)

    conn_rec = nest.GetConnections(source=timer, target=timer)
    conn_ff = nest.GetConnections(source=messenger, target=timer)
    plastic_conns['rec'] = list(conn_rec)
    plastic_conns['ff'] = list(conn_ff)

    return plastic_conns


def _init_recording_structures(parameters, snn, embedded_signal):
    """

    :param parameters:
    :param snn:
    :param embedded_signal:
    :return:
    """
    rec_pars = parameters.recording_pars
    recorded_conns = {}
    n_syn_rec_mean = rec_pars.n_syn_rec_mean if hasattr(rec_pars, 'n_syn_rec_mean') else 500.

    for c_name in rec_pars.record_conn_names:
        # get connections to be recorded
        src_gid = snn.find_population(c_name[1]).gids
        tgt_gid = snn.find_population(c_name[0]).gids
        tmp_conn = nest.GetConnections(source=src_gid, target=tgt_gid)

        # randomize connections to record from if esyn
        if not is_etrace(parameters):
            rand_indices = np.random.choice(np.arange(len(tmp_conn)), min(len(tmp_conn), n_syn_rec_mean), replace=False)
            recorded_conns[c_name] = [tmp_conn[idx] for idx in rand_indices]

        recorded_conns[c_name] = tmp_conn[:n_syn_rec_mean]

    recorded_data = {}
    if rec_pars.samples_to_record:
        for e in rec_pars.samples_to_record.keys():
            recorded_data[e] = {}
            for b in rec_pars.samples_to_record[e].keys():
                recorded_data[e][b] = {
                    'vocabulary': embedded_signal.vocabulary,
                    'batch_seq': None,
                    'batch_size': None,
                    'batch_token_intervals': None,
                    'conn_data': {
                        c_name: {
                            'sampled_times': [],
                            'sampled_hebbian': [],
                            'sampled_ltd': [],
                            'sampled_ltp': [],
                            'sampled_weights': [],
                            'sampled_weights_sum': []
                        } for c_name in rec_pars.record_conn_names
                    },
                    'pop_data': {
                        p: {
                            'sampled_times': [],
                            'sampled_rates': [],
                            'spikelists': []
                        } for p in snn.population_names
                    }
                }

    return recorded_conns, recorded_data


def store_final_weight_matrices(parameters, snn, recorded_conns, storage_paths):
    """
    Stores a dictionary with the final weights of all recorded connections.
    :param parameters:
    :param snn:
    :param recorded_conns:
    :param storage_paths:
    :return:
    """
    W_dict = {}

    for c_name, _ in recorded_conns.items():
        src_gids = snn.find_population(c_name[1]).gids
        tgt_gids = snn.find_population(c_name[0]).gids

        all_conns = nest.GetConnections(src_gids, tgt_gids)
        weights = nest.GetStatus(all_conns, keys=['weight'])
        weights = [w[0] for w in weights]

        W = np.zeros((len(src_gids), len(tgt_gids)))
        min_src_gid = min(src_gids)
        min_tgt_gid = min(tgt_gids)
        for idx in range(len(all_conns)):
            W[all_conns[idx][0] - min_src_gid][all_conns[idx][1] - min_tgt_gid] = weights[idx]

        W_dict[c_name] = W  # store in dictionary

    with open(os.path.join(storage_paths['activity'], f'W_final_{parameters.label}.npy'), 'wb') as f:
        pkl.dump(W_dict, f)


def _randomize_learning_pars(syn_conns_plastic, learning_pars, update_dict):
    """
    Randomizes learning parameters uniformly within a given interval.

    :param syn_conns_plastic:
    :param learning_pars:
    :param update_dict: other synapse parameters that must be set
    :return:
    """
    ru = np.random.uniform
    rand = learning_pars.randomize / 100.

    logging.info(f'Randomizing learning parameters within +-{rand * 100.}% ...')
    for conn in ['rec', 'ff']:
        nest.SetStatus(syn_conns_plastic[conn], [
            copy_dict(update_dict, {
                'tau_ltp': ru(learning_pars['tau_ltp_' + conn] * (1 - rand),
                              learning_pars['tau_ltp_' + conn] * (1 + rand)),
                'tau_ltd': ru(learning_pars['tau_ltd_' + conn] * (1 - rand),
                              learning_pars['tau_ltd_' + conn] * (1 + rand)),
                'eta_ltp': ru(learning_pars['eta_ltp_' + conn] * (1 - rand),
                              learning_pars['eta_ltp_' + conn] * (1 + rand)),
                'eta_ltd': ru(learning_pars['eta_ltd_' + conn] * (1 - rand),
                              learning_pars['eta_ltd_' + conn] * (1 + rand)),
            })
        ])


def _simulate_esyn(parameters, snn, encoder, sampling_interval, reward_times_dict, recorded_data, recorded_conns,
                   epoch, batch, batch_stim_seq, batch_sequence, syn_conns_plastic, storage_paths):
    """
    Processes a single batch sequence for the synapse-centered integration scenario.

    :param parameters:
    :param snn:
    :param encoder:
    :param sampling_interval:
    :param reward_times_dict:
    :param recorded_data:
    :param recorded_conns:
    :param epoch:
    :param batch:
    :param batch_stim_seq:
    :param batch_sequence:
    :param syn_conns_plastic: dictionary {'rec':[], 'ff':[]} with plastic connections
    :return:
    """
    use_plasticity = parameters.connection_pars.plastic_syn_model is not None
    rec_pars = parameters.recording_pars
    # compute reward times
    reward_times = [tup[1] + parameters.learning_pars.reward_delay
                    for idx, tup in enumerate(batch_stim_seq[1]) if batch_sequence[idx] != '#']
    t_syn_update = 0
    # for the test phase, only have a reward at the very end.. it's not needed at all, but at least 1 reward time is req
    if isinstance(batch, str) and 'test' in batch:
        reward_times = reward_times[-1:]
        # set learning rate to 0 to ensure that weights are not modified anymore
        t = time.time()
        nest.SetStatus(syn_conns_plastic['rec'] + syn_conns_plastic['ff'], {'learn_rate': 0.})
        t_syn_update += time.time() - t

    reward_times_dict[epoch][batch] = reward_times
    print(" >>>> reward times: ", reward_times)

    # update synapses with reward times and flush NEST recording arrays in all plastic connections
    if use_plasticity:
        update_dict = {'reward_times': reward_times, 'sampling_interval': 0.}
        t = time.time()
        if 'randomize' not in parameters.learning_pars:
            nest.SetStatus(syn_conns_plastic['rec'] + syn_conns_plastic['ff'], update_dict)
        else:
            _randomize_learning_pars(syn_conns_plastic, parameters.learning_pars, update_dict)
        t_syn_update += time.time() - t

    # mark end of current trial in each neuron to ignore any cross-trial spikes. We need to stop a bit earlier
    # than the actual end in order to safely avoid spikes spilling over between trials. Since we're anyway
    # in the non-stimulated period here, this can be done without issues.
    t_end_trial = batch_stim_seq[0].t_stop - 300.  # TODO we may need to increase 300 here

    # update recording parameters
    if rec_pars.samples_to_record and str(epoch) in rec_pars.samples_to_record \
            and str(batch) in rec_pars.samples_to_record[str(epoch)]:
        print("Initializing recording", flush=True)
        for c_name, conns in recorded_conns.items():
            t = time.time()
            print("\trecord connection: {}".format(c_name), flush=True)
            if use_plasticity:
                nest.SetStatus(conns, {'sampling_interval': sampling_interval})
            t_syn_update += time.time() - t

        # update recording flag in each population and randomize membrane potentials at the beginning of each trial
        for pop_name, pop_obj in snn.populations.items():
            neuron_params = parameters.net_pars.neurons[parameters.net_pars.populations.index(pop_name)]
            t = time.time()
            nest.SetStatus(pop_obj.gids, {
                'logOutput': True, 'rate': 0.,
                'g_ex__X__spikeExcRec0': 0., 'g_ex__X__spikeExcRec1': 0., 'g_in__X__spikeInh': 0.,   # reset conduct.
                'V_m': neuron_params['V_m'],
                't_end_last_trial': t_end_trial,
            })
            t_syn_update += time.time() - t

        # prepare spike recorders for the current interval
        if 'record_spikes_pop' in rec_pars:
            for dev_idx in range(len(snn.device_gids)):
                if snn.device_type[dev_idx][0] == 'spike_detector' and \
                    snn.population_names[dev_idx] in rec_pars['record_spikes_pop']:
                    start_t, stop_t = batch_stim_seq[1][0][0], batch_stim_seq[1][-1][1]
                    nest.SetStatus(snn.device_gids[dev_idx][0], {'start': start_t, 'stop': stop_t})

        t_profile = time.time()
        snn.process_batch('transient', encoder, batch_stim_seq)
        print("Epoch {} Batch {} runtime: {} - recorded".format(epoch, batch, time.time() - t_profile), flush=True)
        # need to simulate everything until the end of the batch (fake spike)
        if use_plasticity:
            t = time.time()
            nest.SetStatus(syn_conns_plastic['rec'] + syn_conns_plastic['ff'],
                           {'fake_spike': nest.GetKernelStatus(keys=['time'])[0]})
            t_syn_update += time.time() - t
            logprint.info(f"[{parameters.label}] Time elapsed for fake-spike: {time.time() - t}")

        recorded_data[str(epoch)][str(batch)]['batch_seq'] = batch_sequence
        recorded_data[str(epoch)][str(batch)]['batch_token_intervals'] = batch_stim_seq[1]
        _extract_recorded_data_esyn(parameters, epoch, batch, snn, recorded_data, recorded_conns, reward_times_dict,
                                    storage_paths)
    else:
        # TODO shouldn't we set the sampling interval back to 0 for this block?
        # update recording flag in each population and randomize membrane potentials at the beginning of each trial
        for pop_name, pop_obj in snn.populations.items():
            neuron_params = parameters.net_pars.neurons[parameters.net_pars.populations.index(pop_name)]
            t = time.time()
            nest.SetStatus(pop_obj.gids, {
                'logOutput': False, 'rate': 0.,
                'g_ex__X__spikeExcRec0': 0., 'g_ex__X__spikeExcRec1': 0., 'g_in__X__spikeInh': 0.,  # reset conduct.
                'V_m': neuron_params['V_m'],
                't_end_last_trial': t_end_trial,
            })
            t_syn_update += time.time() - t

        t_profile = time.time()
        snn.process_batch('transient', encoder, batch_stim_seq)
        print("Epoch {} Batch {} runtime: {} - nonrecorded".format(epoch, batch, time.time() - t_profile), flush=True)

        # need to simulate everything until the end of the batch (fake spike)
        if use_plasticity:
            t = time.time()
            nest.SetStatus(syn_conns_plastic['rec'] + syn_conns_plastic['ff'],
                           {'fake_spike': nest.GetKernelStatus(keys=['time'])[0]})
            t_syn_update += time.time() - t

    logprint.info(f"[{parameters.label}] Time elapsed updating synapses and neurons: {t_syn_update}")


def _extract_recorded_data_esyn(parameters, epoch, batch, snn, recorded_data, recorded_conns, reward_times_dict,
                                storage_paths):
    """
    Extracts data recorded from plastic synapses after each batch, for the ESYN scenario.

    If rec_pars.sampling_interval = 0, query and extract the weights directly from the synapses (not through the
    recording structures within).

    :param epoch:
    :param batch:
    :param snn:
    :param recorded_data:
    :param recorded_conns:
    :param reward_times_dict:
    :return:
    """
    print('Extracting recorded data from epoch #{} and batch #{}'.format(epoch, batch))
    t_elapsed = time.time()
    # avoid code clutter
    use_plasticity = parameters.connection_pars.plastic_syn_model is not None
    batch_data = recorded_data[epoch][batch]
    rec_pars = parameters.recording_pars

    assert len(rec_pars.samples_to_record[epoch][batch]) == 1
    # only retrieve the time window (recorded sample) we are interested inhttps://open.spotify.com/playlist/37i9dQZF1EVJSvZp5AOML2
    seq_to_plot, seq_intervals = input_seq.get_sample_sequence_from_batch(batch_data,
                                                                          rec_pars.samples_to_record[epoch][batch][0])
    t_start_seq = seq_intervals[0][0]
    t_stop_seq = seq_intervals[-1][1]

    if rec_pars.sampling_interval:
        t_start_rec = (int(t_start_seq) // int(rec_pars.sampling_interval)) * rec_pars.sampling_interval  # first valid samp time
        valid_times = np.arange(t_start_rec + rec_pars.sampling_interval, t_stop_seq, rec_pars.sampling_interval)
        times = valid_times
        len_times = len(times)
    else:
        times = np.array([nest.GetKernelStatus(keys=['time'])[0]])
        len_times = len(times)

    for c_name, conns in recorded_conns.items():
        if not use_plasticity:
            continue

        if rec_pars.sampling_interval:
            status_conns = nest.GetStatus(conns, keys=['sampled_times', 'sampled_hebbian', 'sampled_ltp',
                                                       'sampled_ltd', 'sampled_weights'])
            # remove connections for which no data has been recorded (no spikes via that synapse)
            status_conns = [c for c in status_conns if len(c[0])]
        else:
            # only interested in the weights
            status_conns = nest.GetStatus(conns, keys=['w'])
            # remove connections for which no data has been recorded (no spikes via that synapse)
            status_conns = [c for c in status_conns if len(c)]

        if len(status_conns) == 0:
            print(f"Dang! None of the chosen synapses {c_name} have any data recorded... increase recording #synapses?")
            continue

        print('Extracting traces & weights for conn {}, #{} time samples recorded from #{} synapses'.format(
            c_name, len_times, len(status_conns)))

        ltp = np.empty((len(status_conns), len(times)))
        ltd = np.empty((len(status_conns), len(times)))
        weights = np.empty((len(status_conns), len(times)))
        ltp[:] = np.nan
        ltd[:] = np.nan
        weights[:] = np.nan

        time_indices = np.arange(len(times))
        single_traces = []
        plot_counter = parameters.plotting_pars.plot_single_traces  # just to count how many traces to plot

        for c_idx, c in enumerate(status_conns):
            if rec_pars.sampling_interval:
                ltp[c_idx, time_indices] = c[2][-len_times:]
                ltd[c_idx, time_indices] = c[3][-len_times:]
                weights[c_idx, time_indices] = c[4][-len_times:]

                if 'plot_single_traces' in parameters.plotting_pars and \
                        parameters.plotting_pars.plot_single_traces and \
                        plot_counter and np.count_nonzero(ltp[c_idx, :][~np.isnan(ltp[c_idx, :])]) > 0:
                    plot_counter -= 1
                    single_traces.append(c)
            else:
                weights[c_idx] = c[0]

        if rec_pars.sampling_interval:
            plotting.plot_single_traces(single_traces, c_name, batch, parameters, storage_paths)

        # update result for these connections in data to be returned
        if len(times) :
            batch_data['conn_data'][c_name]['sampled_times'].extend(times.tolist())
            batch_data['conn_data'][c_name]['sampled_weights'].extend(np.nanmean(weights, axis=0).tolist())
            batch_data['conn_data'][c_name]['sampled_weights_sum'].extend(np.nansum(weights, axis=0).tolist())
            if rec_pars.sampling_interval:
                batch_data['conn_data'][c_name]['sampled_ltp'].extend(np.nanmean(ltp, axis=0).tolist())
                batch_data['conn_data'][c_name]['sampled_ltd'].extend(np.nanmean(ltd, axis=0).tolist())

    # we need to extract spiking activity here if we are to record it
    if 'record_spikes_pop' in rec_pars:
        snn.extract_activity(flush=True, reset=True)

    for pop_name, pop_obj in snn.populations.items():
        status_data = nest.GetStatus(pop_obj.gids, ['recorded_times', 'recorded_rates'])
        times = status_data[0][0]
        # retrieve estimated (filtered) rates averaged across neurons
        mean_rates = np.mean(np.array([t[1] for t in status_data]), axis=0)

        batch_data['pop_data'][pop_name]['sampled_times'].extend(times)
        batch_data['pop_data'][pop_name]['sampled_rates'].extend(mean_rates.tolist())

        if 'record_spikes_pop' in rec_pars:
            if pop_name in rec_pars['record_spikes_pop']:
                batch_data['pop_data'][pop_name]['spikelists'].append(snn.spiking_activity[snn.population_names.index(pop_name)])
            else:
                batch_data['pop_data'][pop_name]['spikelists'].append(None)

    print('Completed extracting recorded data from epoch #{} and batch #{}, elapsed time: {}'.format(
        epoch, batch, time.time() - t_elapsed))
    return recorded_data


def simulate(parameters, snn, encoder, sequencer, embedded_signal, intervals, storage_paths):
    """
    Main simulation function.

    :param parameters:
    :param snn:
    :param encoder:
    :param sequencer:
    :param embedded_signal:
    :param intervals: [list[float]] contains the symbol-specific ISIs following each token
    :param storage_paths:
    :return:
    """
    rec_pars = parameters.recording_pars
    task_pars = parameters.task_pars

    # whether to sample at regular intervals during trials (slows down simulation significantly)
    sampling_interval = 0. if not hasattr(rec_pars, 'sampling_interval') else rec_pars.sampling_interval
    stim_onset = 1.

    # gather neuron ids in populations having plastic connections
    syn_conns_plastic = _get_plastic_connections(snn)
    logging.info(f"Number of plastic connections: {len(syn_conns_plastic['rec']) + len(syn_conns_plastic['ff'])}")
    print(np.unique(nest.GetStatus(syn_conns_plastic['ff'], ['learn_rate'])))
    assert len(np.unique(nest.GetStatus(syn_conns_plastic['ff'], ['learn_rate']))) == 1

    reward_times_dict = {str(e): {str(b): [] for b in range(task_pars.n_batches)}
                         for e in range(task_pars.n_epochs)}
    reward_times_dict['transient_epoch'] = {'transient_batch': []}  # this is for the transient phase
    reward_times_dict['test_epoch'] = {'test_batch': []}  # this is for the test phase
    recorded_conns, recorded_data = _init_recording_structures(parameters, snn, embedded_signal)

    ###########################################################################################################
    if 'n_batches_transient' in task_pars:
        for batch_trans in range(task_pars.n_batches_transient):
            # transient initial stimulation - single sequence where only the first token is stimulated ################
            batch_sequence = sequencer.string_set[0]
            batch_stim_seq = embedded_signal.draw_stimulus_sequence(batch_sequence, onset_time=stim_onset,
                                                                    continuous=True, intervals=intervals)
            # ensure only the first token is stimulated, set amplitude of rest to 0
            for t in batch_sequence[1:]:
                batch_stim_seq[0].analog_signals[sequencer.tokens.index(t)].signal[:] = 0.
            _simulate_esyn(parameters, snn, encoder, sampling_interval, reward_times_dict, recorded_data,
                           recorded_conns, 'transient_epoch', 'transient_batch', batch_stim_seq, batch_sequence,
                           syn_conns_plastic, storage_paths)
            stim_onset = snn.next_onset_time()

    ###########################################################################################################
    # batch processing - training
    for epoch in range(task_pars.n_epochs):
        for batch in range(task_pars.n_batches):
            t = time.time()
            # for recording, we only want to create batches of size 1
            if rec_pars.samples_to_record and batch in rec_pars.samples_to_record[str(epoch)]:
                batch_sequence = sequencer.string_set[0]
            else:
                batch_sequence = sequencer.generate_sequence(n_strings=task_pars.batch_size)
            batch_stim_seq = embedded_signal.draw_stimulus_sequence(batch_sequence, onset_time=stim_onset,
                                                                    continuous=True, intervals=intervals)

            _simulate_esyn(parameters, snn, encoder, sampling_interval, reward_times_dict, recorded_data,
                           recorded_conns, str(epoch), str(batch), batch_stim_seq, batch_sequence,
                           syn_conns_plastic, storage_paths)

            stim_onset = snn.next_onset_time()
            print("Processing batch {} from epoch {} is fully completed. Overall elapsed time: {} ".format(
                batch, epoch, time.time() - t))

    #####################################################################################
    store_final_weight_matrices(parameters, snn, recorded_conns, storage_paths)
    # testing - single sequence where only the first token is stimulated ################
    for batch in range(task_pars.n_batches_test):
        t = time.time()
        batch_sequence = sequencer.string_set[0]
        batch_stim_seq = embedded_signal.draw_stimulus_sequence(batch_sequence, onset_time=stim_onset,
                                                                continuous=True, intervals=intervals)
        batch_name = f'test_batch_{batch}'  # fixed naming convention, defined in the param file
        # ensure only the first token is stimulated, set amplitude of rest to 0
        for b_idx in batch_sequence[1:]:
            batch_stim_seq[0].analog_signals[sequencer.tokens.index(b_idx)].signal[:] = 0.

        _simulate_esyn(parameters, snn, encoder, sampling_interval, reward_times_dict, recorded_data, recorded_conns,
                       'test_epoch', batch_name, batch_stim_seq, batch_sequence, syn_conns_plastic, storage_paths)

        stim_onset = snn.next_onset_time()
        print("Processing batch {} from epoch {} is fully completed. Overall elapsed time: {} ".format(
            batch, 'test_epoch', time.time() - t))

    snn.extract_activity()
    return reward_times_dict, recorded_data
