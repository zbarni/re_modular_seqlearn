import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.stats as st
from scipy import optimize as opt
import itertools

from fna.tools.network_architect.base import merge_population_activity
from fna.tasks.symbolic.embeddings import VectorEmbeddings
from fna.tasks.preprocessing import ImageFrontend
from fna.tools import visualization as viz
from fna.tools.analysis import metrics as met
from fna.tools.signals.analog import AnalogSignalList
from fna.tools.utils.operations import empty
from fna.decoders.readouts import DummyReadout
from fna.decoders import StateMatrix

from utils.data_handling import convert_performance


def check_aggregator(processor):
    """
    Check whether the aggregator refer to train or test set
    :param processor:
    :return:
    """
    assert isinstance(processor, PostProcessor), "Provide a data aggregate as a PostProcessor object"
    if len(processor.states) == 2:
        indices = [0, -1]
        idx_labels = ['before-train', 'after-train']
    else:
        indices = [-1]
        idx_labels = ['test']

    if 'train' in processor.dataset_label:
        idx_labels = ['train']

    return indices, idx_labels


def plot_inputs(processor, display=False, save=True, plot_input_sample=True, plot_input_space=True,
                plot_embedding_distance=True):
    """
    Simplify the postprocessing chain
    :param processor:
    :return:
    """
    indices, idx_labels = check_aggregator(processor)
    input_signal, _, _, _, _, _ = processor.parse(index=indices[-1], task=None)
    if plot_input_sample:
        processor.analysis_interval = [0., 1000.]
        processor.plot_input_sample(input_signal, label='', display=display, save=save)
    if plot_input_space:
        processor.plot_input_space(display=display, save=save)
    if plot_embedding_distance:
        processor.plot_embedding_distance(display=display, save=save)


def analyse_inputs(processor):
    """

    :param processor:
    :return:
    """
    _ = check_aggregator(processor)
    return {'input_complexity': processor.input_complexity()}


def plot_states(processor, display=False, save=True, plot_mds=False, plot_state_density=False):
    indices, idx_labels = check_aggregator(processor)
    for index, lab in zip(indices, idx_labels):
        _, state_matrix, _, _, label_seq, _ = processor.parse(index=index, task=None)
        processor.analysis_interval = [0., 2000.]
        processor.plot_sample_states(state_matrix, n_neurons=10, label=lab, display=display, save=save)
        # processor.state_distance_distribution(state_matrix, label=lab, display=display, save=save)
        if len(label_seq) < 100 and plot_mds:
            processor.plot_state_space(state_matrix, label=lab, label_seq=label_seq, metric='MDS', cmap='jet',
                                       display=display, save=save)

        if plot_state_density:
            processor.plot_state_density(state_matrix, label=lab, display=display, save=save)


def analyse_states(processor):
    indices, idx_labels = check_aggregator(processor)
    result = {}
    for index, lab in zip(indices, idx_labels):
        _, state_matrix, _, _, _, _ = processor.parse(index=index, task=None)
        result.update({'state_complexity-{}'.format(lab): processor.state_complexity(state_matrix)})
        if state_matrix.matrix.shape[0] > 3:
            eff_d = met.compute_dimensionality(state_matrix.matrix, pca_obj=None, label=lab, plot=False,
                                       display=False, save=False)
            result.update({'state_dimensionality-{}'.format(lab): eff_d})
    return result


def state_stats(processor, analysis_pars, display=False, save=True):
    indices, idx_labels = check_aggregator(processor)
    result = {}
    for index, lab in zip(indices, idx_labels):
        _, state_matrix, _, _, _, _ = processor.parse(index=index, task=None)
        rates, cvs, ccs = processor.state_stats(state_matrix.matrix, analysis_pars.population_activity['n_bins'],
                                                analysis_pars.population_activity['n_pairs'], display=display,
                                                save=save)
        result.update({'state_averages-{}'.format(lab): rates})
        result.update({'state_cvs-{}'.format(lab): cvs})
        result.update({'state_ccs-{}'.format(lab): ccs})
    return result


def state_timescale(processor, max_lag=1000, display=False, save=True):
    indices, idx_labels = check_aggregator(processor)
    result = {}
    for index, lab in zip(indices, idx_labels):
        _, state_matrix, _, _, _, _ = processor.parse(index=index, task=None)
        final_acc, mean_fit, acc_function, time_scales = processor.state_timescale(state_matrix, max_lag, display, save)
        result.update({'state_tau-int-{}'.format(lab): time_scales})
    return result


def plot_accuracies(processor, tasks, symbolic=False, metrics=None, display=False, save=True):
    indices, idx_labels = check_aggregator(processor)
    ctr = 0
    for task in tasks:
        for index, lab in zip(indices, idx_labels):
            if symbolic:
                _, _, internal_output, internal_target, _, decoders = processor.parse(index=index, task=task)
            else:
                assert False
                _, _, internal_output, internal_target, _, decoders = processor.parse(index=index, task=task)
            if decoders is not None:
                dec = [x[0] for x in decoders]
            else:
                dec = []
            if len(idx_labels) == 2 and ctr == 0:
                processor.plot_training_loss(dec, label=task, display=display, save=save)
            if metrics is not None and isinstance(metrics, list):
                for mt in metrics:
                    processor.plot_decoder_accuracy(internal_output, internal_target, symbolic=symbolic,
                                                    metric=mt, display=display, save=save)
            else:
                processor.plot_decoder_accuracy(internal_output, internal_target, symbolic=symbolic,
                                                metric=metrics, display=display, save=save)


def plot_outputs(processor, tasks, symbolic=False, display=False, save=True):
    indices, idx_labels = check_aggregator(processor)
    for task in tasks:
        for index, lab in zip(indices, idx_labels):
            _, _, internal_output, internal_target, _, decoders = processor.parse(index=index, task=task)
            if decoders is not None:
                dec = [x[0] for x in decoders]
                dec_acc = [x.performance['raw']['MSE'] for x in dec]
                dec_idx = [x[1] for x in decoders]
            else:
                dec = []
                dec_acc = []
                dec_idx = []

            if lab == 'train':
                decoder_outputs = [processor.decoder_outputs[index][x] for x in dec_idx]
                decoder_targets = [processor.decoder_targets[index][x] for x in dec_idx]
            else:
                decoder_outputs = [processor.decoder_outputs[x] for x in dec_idx]
                decoder_targets = [processor.decoder_targets[x] for x in dec_idx]
            # if len(indices) == 2:
            #     decoder_outputs = [processor.decoder_outputs[index][x] for x in dec_idx]
            #     decoder_targets = [processor.decoder_targets[index][x] for x in dec_idx]
            # else:
            #     decoder_outputs = [processor.decoder_outputs[x] for x in dec_idx]
            #     decoder_targets = [processor.decoder_targets[x] for x in dec_idx]

            # add internal output
            r0 = DummyReadout(processor.network_label+lab, internal_output, internal_target)
            acc = r0.evaluate(symbolic=symbolic)
            dec.insert(0, r0)
            dec_acc.insert(0, acc['raw']['MSE'])
            decoder_outputs.insert(0, r0.output)
            decoder_targets.insert(0, r0.test_target)

            processor.plot_outputs_extended(dec, decoder_outputs, decoder_targets, dec_acc, label='{}-{}'.format(task, lab),
                                            display=display, save=save)


def analyse_outputs(processor, tasks):
    indices, idx_labels = check_aggregator(processor)
    result = {}
    for task in tasks:
        for index, lab in zip(indices, idx_labels):
            _, _, internal_output, internal_target, _, decoders = processor.parse(index=index, task=task)
            out_compl, target_compl = processor.output_complexity(internal_output, internal_target)
            result.update({'output_complexity-{}-{}'.format(task, lab): out_compl})
        result.update({'target_complexity-{}'.format(task): target_compl})
    return result


def transfer_functions(processor, amplitude_range, steps, times, network=None, plot=True, display=False, save=False):
    result = {}
    inputs = np.linspace(amplitude_range[0], amplitude_range[1], steps)
    store = processor.storage_paths['figures'] + processor.storage_paths['label']

    if processor.network_type == 'SpikingNetwork':
        r1 = processor.spk_transfer_functions(network, amplitude_range, steps, times, plot=plot, display=display,
                                              save=save)
        result.update(r1)
    indices, idx_labels = check_aggregator(processor)
    for index, lab in zip(indices, idx_labels):
        _, state_matrix, _, _, _, _ = processor.parse(index=index, task=None)

        if state_matrix.matrix.shape[1] != inputs.shape[0]:
            print("Incorrect shape for state matrix {}, please take a single state sample per step, "
                  "ignoring".format(lab))
            continue
        else:
            state_outputs = state_matrix.matrix.mean(0)
            result.update({'state_outputs': state_outputs,
                           'state_io_properties': io_curve_stats(inputs, state_outputs,
                                                                 label=processor.storage_paths['label']+'-states',
                                                                 unit='', verbose=True)})
            if plot:
                fig, ax = plt.subplots(figsize=(4, 4))
                met.plot_io_curve(inputs, state_outputs, ax=ax, save=False, display=False, **{'linewidth': 2, 'color': 'k',
                                                                                  'label': 'state'})
                if save:
                    viz.helper.fig_output(fig, display=display, save=store+'-state-transfer-function.pdf')
                else:
                    viz.helper.fig_output(fig, display=display, save=save)
    return result


# noinspection PyTupleAssignmentBalance
def sustaining_activity_lifetime(processor, network, analysis_interval, input_off_time, input_off_step, verbose=True,
                                 plot=True, save=True):

    def acc_function(x, a, b, tau):
        return a * (np.exp(-x / tau) + b)

    store = processor.storage_paths['figures'] + processor.storage_paths['label']
    result = {}

    if processor.network_type == 'SpikingNetwork':
        r1 = met.ssa_lifetime(network, analysis_interval, input_off=input_off_time, display=verbose)
        result.update({'spiking-ssa': r1})

    indices, idx_labels = check_aggregator(processor)
    for index, lab in zip(indices, idx_labels):
        _, state_matrix, _, _, _, _ = processor.parse(index=index, task=None)

        sm = state_matrix.matrix

        if plot:
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(sm.mean(0), 'k', lw=2)
            ax.plot(np.diff(sm.mean(0)), 'k', lw=2, alpha=0.8)
            ax.vlines(input_off_step, sm.min(), sm.max(), lw=2)
            ax.set_xlim([0., len(sm.mean(0))])
            ax.set_ylim([sm.min(), sm.max()])

        ssm = sm[:, int(input_off_step):]
        acc = np.zeros((ssm.shape[0], ssm.shape[1]-1))
        for iidx in range(ssm.shape[0]):
            acc[iidx, :] = np.diff(ssm[iidx, :])

        time_axis = np.arange(acc.shape[1])
        time_scales = []
        final_acc = []
        errors = []
        initial_guess = np.array([1., 0., 10.])
        for n_signal in range(acc.shape[0]):
            try:
                fit, _ = opt.leastsq(met.err_func, initial_guess, args=(time_axis, acc[n_signal, :], acc_function))

                if fit[2] > 0.1:
                    error_rates = np.sum((acc[n_signal, :] - acc_function(time_axis[:], *fit)) ** 2)
                    print("Lifetime [SSA] = {0} ms / error = {1}".format(str(fit[2]), str(error_rates)))
                    time_scales.append(fit[2])
                    errors.append(error_rates)
                    final_acc.append(acc[n_signal, :])
                elif 0. < fit[2] < 0.1:
                    fit, _ = opt.leastsq(met.err_func, initial_guess, args=(time_axis[1:], acc[n_signal, 1:],
                                                                            acc_function))
                    error_rates = np.sum((acc[n_signal, :] - acc_function(time_axis[:], *fit)) ** 2)
                    print("Timescale [ACC] = {0} ms / error = {1}".format(str(fit[2]), str(error_rates)))
                    time_scales.append(fit[2])
                    errors.append(error_rates)
                    final_acc.append(acc[n_signal, :])
            except:
                continue
        final_acc = np.array(final_acc)

        mean_fit, _ = opt.leastsq(met.err_func, initial_guess, args=(time_axis, np.mean(final_acc, 0), acc_function))

        if mean_fit[2] < 0.1:
            mean_fit, _ = opt.leastsq(met.err_func, initial_guess, args=(time_axis[1:], np.mean(final_acc, 0)[1:],
                                                                         acc_function))
        error_rates = np.sum((np.mean(final_acc, 0) - acc_function(time_axis, *mean_fit)) ** 2)
        print("*******************************************")
        print("Timescale [SSA] = {0} ms / error = {1}".format(str(mean_fit[2]), str(error_rates)))
        print("Accepted dimensions = {0}".format(str(float(final_acc.shape[0]) / float(acc.shape[0]))))

        result.update({'state-ssa-{}'.format(lab): time_scales})

    if plot:
        if save:
            viz.plotting.plot_acc(time_axis, acc, mean_fit, acc_function, ax=None, display=verbose, save=store+'-SSA')
        else:
            viz.plotting.plot_acc(time_axis, acc, mean_fit, acc_function, ax=None, display=verbose, save=store)

    return result


def io_curve_stats(input, output, label, unit, verbose=True):
    """
    Determine min, max, slope ...
    :param input:
    :param output:
    :return:
    """
    if not isinstance(input, np.ndarray):
        input = np.array(input)
    if not isinstance(output, np.ndarray):
        output = np.array(output)

    idx = np.min(np.where(output))

    result = {'min': np.min(output[output > 0.]),
              'max': np.max(output[output > 0.])}

    iddxs = np.where(output)
    slope, intercept, r_value, p_value, std_err = st.linregress(input[iddxs], output[iddxs])
    if "rate" in label:
        slope *= 1000.
    result.update({'slope': slope, 'zero-cross': [input[idx - 1], input[idx]]})

    if verbose:
        print("Output range for {} = [{}, {}] {}".format(label, str(result['min']), str(result['max']), unit))
        print("Critical input for {} in [{}, {}]".format(label, str(input[idx - 1]), str(input[idx])))
        print("Slope for {} = {} {} [linreg method]".format(label, slope, unit))
    return result


# ######################################################################################################################
class PostProcessor(object):
    def __init__(self, network_type, network_label, decoders, experiment_parameters, data_batch, results,
                 sequencer=None, embedding=None, input_signal=None, dataset_label="", storage_paths={}):
        """
        PostProcessor constructor - parse data and run standard processing routines
        :param net:
        :param experiment_parameters:
        :param data_batch:
        :param results:
        :param sequencer:
        :param embedding:
        :param input_signal:
        :param dataset_label:
        :param task_label:
        :param storage_paths:
        """
        self.parameters = experiment_parameters
        self.time_axis = None
        self.analysis_interval = None
        self.inputs = data_batch['inputs']
        self.input_signal = input_signal

        self.decoders = decoders
        if results is not None:
            if not empty(results['decoder_accuracy']):
                self.decoder_outputs, self.decoder_targets, self.decoder_accuracy = \
                    results['decoder_outputs'], results['decoder_targets'], results['decoder_accuracy']
                if 'decoder_loss' in results.keys(): # training results data
                    self.decoder_losses = results['decoder_loss']
                    self.losses = results['losses']
            else: # no decoders used
                self.decoder_outputs, self.decoder_targets, self.decoder_accuracy, self.decoder_losses, self.losses = \
                    None, None, None, None, None
            self.states, self.internal_outputs = results['states'], results['outputs']
        else:
            self.decoder_outputs, self.decoder_targets, self.decoder_accuracy, self.decoder_losses, self.losses, \
            self.states, self.internal_outputs = None, None, None, None, None, None, None

        self.decoder_target_out = data_batch['decoder_outputs']
        if "targets" in data_batch.keys():
            self.internal_targets = data_batch['targets']
        else:
            self.internal_targets = None

        self.state_distances = None
        self.dataset_label = dataset_label
        self.sequencer = sequencer
        self.embedding = embedding
        self.storage_paths = storage_paths
        self.sample_sequence = sequencer.generate_random_sequence(T=50, verbose=False)
        self.network_type = network_type
        self.network_label = network_label

        if input_signal is not None:
            self.input_dimensions = input_signal.embedding_dimensions
        else:
            self.input_dimensions = embedding.embedding_dimensions

    def parse(self, index=0, task=None):
        """
        Extract relevant data for plotting
        :param index: determines whether to analyse the first or last set of data
         (for training results, otherwise it doesn't matter)
        :return:
        """
        assert index in [0, -1], "Index must be 0 or -1"

        if self.decoders is not None:
            decoder_set = self._parse_decoders(task)
        else:
            decoder_set = None

        if self.network_type == "cRNN" or self.network_type == "rRNN" or self.network_type == "ContinuousRateRNN" or \
                self.network_type == "ReservoirRecurrentNetwork":
            self.time_axis = np.arange(0., self.parameters.input_pars.unfolding['duration'] *
                                  self.parameters.task_pars.batch_size,
                                  self.parameters.input_pars.unfolding['dt'])
            input_signal = self.inputs[:, index, :].T
            if self.states is not None:
                if isinstance(self.states, list):
                    state_matrix = self.states[index]
                else:
                    # state_matrix = self.states[index] @zbarni
                    state_matrix = self.states

                if self.network_type == "ReservoirRecurrentNetwork":
                    state_matrix = state_matrix
                else:
                    if isinstance(state_matrix, list):
                        state_matrix = state_matrix[index][:, -1, :].T
                    else:
                        if len(state_matrix.shape) == 2:
                            state_matrix = state_matrix.T
                        else:
                            state_matrix = state_matrix[:, index, :].T
                sm = StateMatrix(state_matrix, label=self.dataset_label, state_var="internal",
                                 population=self.network_label)
            else:
                sm = None
            if self.internal_outputs is not None: # TODO - correct internal output (sub-sample)
                if len(self.internal_outputs[index].shape) == 2:
                    output = self.internal_outputs[index].T
                elif len(self.internal_outputs[index].shape) == 1:
                    output = self.internal_outputs[index]
                else:
                    output = self.internal_outputs[index][:, -1, :].T
            else:
                output = None

            target_seq = self.decoder_target_out[index][0]['output']
            # out_signal = VectorEmbeddings(vocabulary=np.unique(target_seq)).one_hot()
            out_signal = VectorEmbeddings(vocabulary=self.embedding.vocabulary).one_hot()
            target = out_signal.draw_stimulus_sequence(target_seq, as_array=True, verbose=False)
            self.analysis_interval = [self.time_axis.min(), self.time_axis.max()]

        else:
            raise NotImplementedError("Network type not understood")

        return input_signal, sm, output, target, target_seq, decoder_set

    def _parse_decoders(self, task):
        """
        Extract only the readouts for a specific task
        :param task:
        :return:
        """
        if task is not None:
            readouts = [(x, idx) for idx, x in enumerate(self.decoders.readouts) if x.task == task]
            assert not empty(readouts), "No readouts for task {} found".format(task)
            return readouts
        else:
            return [(x, idx) for idx, x in enumerate(self.decoders.readouts)]

    def _state_distance(self, state_matrix):
        if not isinstance(state_matrix, StateMatrix):
            sm = StateMatrix(state_matrix, label=self.dataset_label, state_var="", population=self.network_label)
            state_matrix = sm
        self.state_distances = met.pairwise_dist(state_matrix.matrix)

    def plot_input_sample(self, input_signal, label='', display=False, save=False):
        """
        Plot a sample of the input signals
        :param input_signal:
        :param label:
        :param display:
        :param save:
        :return:
        """
        store = self.storage_paths['figures'] + self.storage_paths['label']
        # if self.embedding is not None:
        #     if not save:
        #         self.embedding.plot_sample(self.sample_sequence, save=store)
        #     else:
        #         self.embedding.plot_sample(self.sample_sequence, save=store+label+'-stim_seq.pdf')

        if self.input_dimensions > 10:
            input_idx = np.random.permutation(self.input_dimensions)[:10]
            input_dimensions = 10
        else:
            input_idx = np.arange(self.input_dimensions)
            input_dimensions = self.input_dimensions

        plt.close('all')
        fig1, axes = plt.subplots(input_dimensions, 1, sharex=True, figsize=(10, 2*input_dimensions))

        if not isinstance(axes, np.ndarray):
            axes = [axes]
        for idx, (neuron, ax) in enumerate(zip(input_idx, axes)):
            ax.plot(self.time_axis, input_signal[neuron, :], 'k')
            ax.set_ylabel(r"$U_{"+"{0}".format(neuron)+"}$")
            ax.set_xlim(self.analysis_interval)
            if idx == input_dimensions - 1:
                ax.set_xlabel("Time [ms]")
        if save:
            viz.helper.fig_output(fig1, display, self.storage_paths['figures']+self.storage_paths['label']+
                                  label+'-inputs.pdf')
        else:
            viz.helper.fig_output(fig1, display, self.storage_paths['figures']+self.storage_paths['label'])

    def input_complexity(self):
        unique_seq = self.embedding.draw_stimulus_sequence(self.sequencer.tokens, as_array=True, verbose=False)
        input_d = met.pairwise_dist(unique_seq.T)
        embedding_complexity = (np.mean(input_d[input_d != 0]), np.std(input_d[input_d != 0]))
        return {'sequence_entropy': self.sequencer.entropy(self.sample_sequence),
                'sequence_compressibility': self.sequencer.compressibility(self.sample_sequence),
                'embedding_complexity': embedding_complexity}

    def plot_input_space(self, display=False, save=False):
        store = self.storage_paths['figures']+self.storage_paths['label']

        # plot input space
        input_mat = self.embedding.draw_stimulus_sequence(self.sample_sequence, as_array=True, verbose=False)
        if save:
            viz.plotting.plot_discrete_space(input_mat, data_label=self.dataset_label, label_seq=self.sample_sequence,
                                             metric='MDS', colormap='jet', display=display,
                                             save=store+"-input-space")
        else:
            viz.plotting.plot_discrete_space(input_mat, data_label=self.dataset_label, label_seq=self.sample_sequence,
                                             metric='MDS', colormap='jet', display=display, save=store)

    def plot_embedding_distance(self, display=False, save=False):
        store = self.storage_paths['figures'] + self.storage_paths['label']
        # pairwise distances between stimuli
        unique_seq = self.embedding.draw_stimulus_sequence(self.sequencer.tokens, as_array=True, verbose=False)
        input_d = met.pairwise_dist(unique_seq.T)
        if save:
            fig, ax = viz.plotting.plot_matrix(input_d, labels=self.sequencer.tokens,
                                               data_label=self.dataset_label+"-embedding-distances", display=display,
                                               save=store+"-input-pairwise-dist.pdf")
        else:
            fig, ax = viz.plotting.plot_matrix(input_d, labels=self.sequencer.tokens,
                                               data_label=self.dataset_label+"-embedding-distances", display=display,
                                               save=store)

    def plot_sample_states(self, state_matrix, n_neurons=10, label='', display=False, save=False):
        store = self.storage_paths['figures'] + self.storage_paths['label']

        # State profiler
        if self.state_distances is None:
            self._state_distance(state_matrix)

        if save:
            # state_matrix.plot_sample_traces(n_neurons=n_neurons, display=display,
            #                                 save=store+'-{}-state-matrix-trace.pdf'.format(label))
            state_matrix.plot_matrix(display=display, save=store+'-{}-states.pdf'.format(label))
            # state_matrix.plot_trajectory(display=display, save=store+'-{}-trajectory.pdf'.format(label))
        else:
            state_matrix.plot_sample_traces(n_neurons=n_neurons, display=display, save=store)
            state_matrix.plot_matrix(display=display, save=store)
            state_matrix.plot_trajectory(display=display, save=store)

        # if save:
        #     _ = viz.plotting.plot_matrix(self.state_distances, labels=None,
        #                                        data_label=self.dataset_label+"-{}-state-distances".format(label),
        #                                        display=display, save=store+'-{}-state-distances.pdf'.format(label))
        # else:
        #     _ = viz.plotting.plot_matrix(self.state_distances, labels=None,
        #                                        data_label=self.dataset_label+"-{}-state-distances".format(label),
        #                                        display=display, save=store)

    def state_complexity(self, state_matrix):
        """
        Quantify the complexity of the state matrix
        (currently as mean pairwise Euclidean distance)
        :return:
        """
        if self.state_distances is None:
            self._state_distance(state_matrix)
        return (np.mean(self.state_distances[self.state_distances != 0]),
                np.std(self.state_distances[self.state_distances != 0]))

    def state_distance_distribution(self, state_matrix, label='', display=False, save=False):
        store = self.storage_paths['figures'] + self.storage_paths['label']
        if self.state_distances is None:
            self._state_distance(state_matrix)
        if save:
            viz.plotting.plot_histogram(self.state_distances[self.state_distances != 0],
                                        n_bins=100, norm=True, mark_mean=True, display=display,
                                        save=store+'-{}-state-pdists.pdf'.format(label),
                                        **{'suptitle': self.dataset_label+"{}-pairwise-state-distances".format(label),
                                           'xlabel': r"$||X_{i}, X_{j}||_{2}$", 'ylabel': "Frequency"})
        else:
            viz.plotting.plot_histogram(self.state_distances[self.state_distances != 0],
                                        n_bins=100, norm=True, mark_mean=True, display=display, save=store,
                                        **{'suptitle': self.dataset_label+"{}-pairwise-state-distances".format(label),
                                           'xlabel': r"$||X_{i}, X_{j}||_{2}$", 'ylabel': "Frequency"})

    def plot_state_space(self, state_matrix, label='', label_seq=None, metric='MDS', cmap='jet', display=False,
                         save=False):
        store = self.storage_paths['figures'] + self.storage_paths['label']
        if self.state_distances is None:
            self._state_distance(state_matrix)
        if save:
            viz.plotting.plot_discrete_space(state_matrix.matrix, data_label='', label_seq=label_seq, metric=metric,
                                             colormap=cmap, display=display, save=store+'-{}-states-scatter'.format(label))
        else:
            viz.plotting.plot_discrete_space(state_matrix.matrix, data_label=label, label_seq=label_seq, metric=metric,
                                             colormap=cmap, display=display, save=store)

    def plot_state_density(self, state_matrix, label='', cmap="viridis", display=False, save=False):
        store = self.storage_paths['figures'] + self.storage_paths['label']
        if self.state_distances is None:
            self._state_distance(state_matrix)

        # density of first 2 PCs
        pca_obj = PCA(n_components=2)
        X_r = pca_obj.fit(state_matrix.matrix.T).transform(state_matrix.matrix.T).T
        xy = np.vstack([X_r[0, :], X_r[1, :]])
        z = st.gaussian_kde(xy)(xy)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle("{} - PC density".format(self.dataset_label+label))
        scatter1 = ax.scatter(X_r[0, :], X_r[1, :], c=z, s=50, edgecolors=None, cmap=cmap)
        fig.colorbar(scatter1, ax=ax)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.tight_layout()
        if save:
            viz.helper.fig_output(fig, display, save=store+"-{}-pc-density.pdf".format(label))
        else:
            viz.helper.fig_output(fig, display, save=store)

    @staticmethod
    def output_complexity(output, target):
        out_dist = met.pairwise_dist(output.T)
        targ_dist = met.pairwise_dist(target.T)

        return (np.mean(out_dist[out_dist != 0.]), np.std(out_dist[out_dist != 0.])), \
               (np.mean(targ_dist[targ_dist != 0.]), np.std(targ_dist[targ_dist != 0.]))


    def plot_outputs(self, decoders, outputs, targets, accuracies, label='', display=False,
                     save=False):
        store = self.storage_paths['figures'] + self.storage_paths['label']

        # outputs and targets
        fig1, axes = plt.subplots(len(decoders), 1, sharex=True, figsize=(12, 3*len(decoders)))
        fig1.suptitle("{}-{} - Example Outputs".format(self.dataset_label, label))
        if len(outputs) == 1:
            axes = [axes]

        for idx, (out, target, acc) in enumerate(zip(outputs, targets, accuracies)):
            ax = axes[idx]
            ax.set_title(r'MSE={}'.format(acc))

            if len(out.shape) == 1:
                out = out.reshape(-1, 1).T

            if len(target.shape) == 1:
                target = target.reshape(-1, 1).T

            ax.plot(out[0, :], 'k', label="{} - {}".format(decoders[idx].label, label))
            ax.plot(target[0, :], 'r', label='target')
            ax.set_xlim([0, len(target[0, :])])
            ax.legend()
        plt.legend()
        if save:
            viz.helper.fig_output(fig1, display, save=store+"-{}-decoded-outputs.pdf".format(label))
        else:
            viz.helper.fig_output(fig1, display, save=store)

    def plot_outputs_extended(self, decoders, outputs, targets, accuracies, label='', display=False, save=False):
        store = self.storage_paths['figures'] + self.storage_paths['label']
        n_stim = self.input_dimensions

        fig1, axes = plt.subplots(n_stim, len(decoders), sharex=True, figsize=(20, 15))
        fig1.suptitle("{}-{} - Example Outputs".format(self.dataset_label, label))
        if len(outputs) == 1:
            axes = [axes]

        for idx, (out, target, acc) in enumerate(zip(outputs, targets, accuracies)):
            axes[0, idx].set_title(r'MSE={}'.format(acc))

            if len(out.shape) == 1:
                out = out.reshape(-1, 1).T

            if len(target.shape) == 1:
                target = target.reshape(-1, 1).T

            for ch in range(n_stim):
                label_out = "{} - {}".format(decoders[idx].label, label) if ch == 0 else None
                label_tgt = 'target' if ch == 0 else None
                axes[ch, idx].plot(out[ch, :], 'k', label=label_out)
                axes[ch, idx].plot(target[ch, :], 'r', label=label_tgt)
                axes[ch, idx].set_xlim([0, len(target[0, :])])
            axes[0, idx].legend()
        plt.legend()
        if save:
            viz.helper.fig_output(fig1, display, save=store+"-{}-decoded-outputs.pdf".format(label))
        else:
            viz.helper.fig_output(fig1, display, save=store)

    def plot_decoder_accuracy(self, internal_output, internal_target, metric="raw-MSE", symbolic=True,
                              display=False, save=False):

        store = self.storage_paths['figures'] + self.storage_paths['label']
        decoder_accuracy_df = self.decoder_accuracy

        if self.network_type not in ["SpikingNetwork", "SNN"]:
            # add internal output results
            r0 = DummyReadout(self.network_label, internal_output, internal_target)
            acc = r0.evaluate(symbolic=symbolic, vocabulary=self.embedding.vocabulary)
            internal_acc = convert_performance(readout=r0, performance_dict=acc)
            if decoder_accuracy_df is not None and not internal_acc.empty:
                decoder_accuracy_df = decoder_accuracy_df.merge(internal_acc, left_index=True, right_index=True)
            elif not internal_acc.empty:
                decoder_accuracy_df = internal_acc

        # remove underscores from decoder_accuracy df
        change_indices = {}
        for x in decoder_accuracy_df.index:
            if len(x.split("_")) > 1:
                change_indices.update({x: "-".join(x.split("_"))})
        if not empty(change_indices):
            decoder_accuracy_df = decoder_accuracy_df.rename(index=change_indices)
        if metric in change_indices.keys():
            metric = change_indices[metric]

        if metric is not None:
            # if self.dataset_label is not None:
            #     lab = '{}-'.format(self.dataset_label) + metric
            # else:
            #     lab = metric
            lab = metric

            ax = decoder_accuracy_df.loc[metric].plot.bar(rot=45)
            fig, ax = viz.helper.check_axis(ax)
            fig.suptitle(lab)
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)
            ax.set_ylabel(metric)
            if save:
                viz.helper.fig_output(fig, display, save=store+"-decoder-performance-{}.pdf".format(lab))
            else:
                viz.helper.fig_output(fig, display, save=store)
        else:
            ax = decoder_accuracy_df.T.plot.bar(rot=45, subplots=True, figsize=(15, 15), title=self.dataset_label)
            fig, ax = viz.helper.check_axis(ax[0])
            # fig.suptitle(self.dataset_label, y=1.)
            # fig.subplots_adjust(hspace=0.8)
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)
            if save:
                # viz.helper.fig_output(fig, display, save=store+"-decoder-performance-{}.pdf".format(self.dataset_label))
                viz.helper.fig_output(fig, display, save=store+"-decoder-performance.pdf")
            else:
                viz.helper.fig_output(fig, display, save=store)

    def plot_training_loss(self, decoders, label, cmap="cividis", display=False, save=False):
        store = self.storage_paths['figures'] + self.storage_paths['label']
        readout_labels = [x.label+'-'+x.task for x in decoders]
        r_colors = viz.helper.get_cmap(len(readout_labels), cmap)
        n_epochs = self.parameters.task_pars.n_epochs

        # Training Loss
        totals = len(readout_labels) + 1
        fig, ax = plt.subplots(1, totals, figsize=(int(5*totals), 4))
        fig.suptitle("Training Loss - {}".format(label))
        x_ticks = np.arange(n_epochs)+1

        if not empty(self.losses):
            ax[0].plot(np.arange(n_epochs)+1, self.losses, 'k', label="internal")
            ax[0].set_xticks(x_ticks[::10]-1)
            ax[0].set_xlim([x_ticks.min(), x_ticks.max()])
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("MSE")
            ax[0].legend()

        if not empty(self.decoder_losses):
            decoder_losses = {k: [] for k in readout_labels}
            for batch, losses in self.decoder_losses.items():
                for r_label in readout_labels:
                    decoder_losses[r_label].append(losses[r_label]['raw-MSE'])
            for idx, (k, v) in enumerate(decoder_losses.items()):
                lab = '-'.join(k.split('-')[:-1])
                ax[idx+1].plot(np.arange(n_epochs)+1, v, color=r_colors(idx), label=lab)
                ax[idx+1].set_xticks(x_ticks[::10]-1)
                ax[idx+1].set_xlim([x_ticks.min(), x_ticks.max()])
                ax[idx+1].set_xlabel("Epoch")
                ax[idx+1].set_ylabel("MSE")
                ax[idx+1].legend()

        if save:
            viz.helper.fig_output(fig, display, save=store+"-training-loss-{}.pdf".format(label))
        else:
            viz.helper.fig_output(fig, display, save=store)

    def state_stats(self, state_matrix, bins, n_pairs, cmap="viridis", display=False, save=False):
        store = self.storage_paths['figures'] + self.storage_paths['label']
        cvs = [state_matrix[i, :].std()/state_matrix[i, :].mean() for i in range(state_matrix.shape[0])]
        rates = state_matrix.mean(0)
        ccs = []
        for ii in range(n_pairs):
            i, j = np.random.choice(state_matrix.shape[0]), np.random.choice(state_matrix.shape[0])
            ccs.append(np.corrcoef(state_matrix[i, :], state_matrix[j, :])[0, 1])

        fig, axes = plt.subplots(1, 3, figsize=(13, 5))
        axes_pr = [{'title': r"Mean activity $\bar{X}$"}, {'title': r"$\mathrm{CV(x)}$"}, {'title': r"$\mathrm{CC(x)}$"}]
        fig.suptitle(self.dataset_label+'-State statistics')

        if save:
            viz.plotting.plot_histograms(axes, [rates, cvs, ccs], [bins, bins, bins], args_list=axes_pr, cmap=cmap,
                                         kde=False, display=display, save=store+"-state-stats.pdf")
        else:
            viz.plotting.plot_histograms(axes, [rates, cvs, ccs], [bins, bins, bins], args_list=axes_pr, cmap=cmap,
                                         kde=False, display=display, save=save)
        return rates, cvs, ccs

    def state_timescale(self, state_matrix, max_lag, display=False, save=False):
        store = self.storage_paths['figures'] + self.storage_paths['label']
        if save:
            final_acc, mean_fit, acc_function, time_scales = met.compute_timescale(state_matrix.matrix, self.time_axis,
                                                                                   max_lag=max_lag, method=0, plot=True,
                                                                                   display=display,
                                                                                   save=store + "-tau-int.pdf")
        else:
            final_acc, mean_fit, acc_function, time_scales = met.compute_timescale(state_matrix.matrix, self.time_axis,
                                                                                   max_lag=max_lag, method=0, plot=True,
                                                                                   display=display,
                                                                                   save=store)
        return final_acc, mean_fit, acc_function, time_scales

    def spiking_stats(self, network, parameters, epochs=None, colormap='viridis', analysis_interval=None,
                      plot=True, display=False, save=True):
        store = self.storage_paths['figures'] + self.storage_paths['label']
        if analysis_interval is None:
            analysis_interval = self.analysis_interval

        if save:
            res = met.characterize_population_activity(network, parameters, analysis_interval, epochs=epochs,
                                                 color_map=colormap, plot=plot, display=display,
                                                 save=store + self.dataset_label,
                                                 color_subpop=True, analysis_pars=parameters.analysis_pars)
        else:
            res = met.characterize_population_activity(network, parameters, analysis_interval, epochs=epochs,
                                                 color_map=colormap, plot=plot, display=display, save=store,
                                                 color_subpop=True, analysis_pars=parameters.analysis_pars)
        return res

    def spk_transfer_functions(self, network, amplitude_range, steps, times, plot=True, display=False, save=True):
        inputs = np.linspace(amplitude_range[0], amplitude_range[1], steps)
        store = self.storage_paths['figures'] + self.storage_paths['label']

        if self.network_type == 'SpikingNetwork':
            analysis_intervals = [[times[idx], times[idx+1]] for idx in range(len(times)-1)]
            spike_lists = {k: v.spiking_activity for k, v in network.populations.items()}
            merged_activity = merge_population_activity(network.merged_populations, start=0., stop=max(times),
                                                        in_place=True)
            spike_lists.update({k: v['spiking_activity'] for k, v in merged_activity.items()})
            mean_rates, corrected_rates, cvs, ents, isi_5ps, result = {}, {}, {}, {}, {}, {}
            for k, v in spike_lists.items():
                mean_rates.update({k: []})
                corrected_rates.update({k: []})
                cvs.update({k: []})
                ents.update({k: []})
                isi_5ps.update({k: []})
                result.update({k: {}})
                for interval in analysis_intervals:
                    sl = v.time_slice(interval[0], interval[1])
                    r1 = met.compute_spike_stats(sl, time_bin=interval[1]-interval[0], summary_only=True, display=False)
                    r2 = met.compute_isi_stats(sl, summary_only=True)

                    corrected_rates[k].append(r1['corrected_rates'][0])
                    mean_rates[k].append(r1['mean_rates'][0])
                    cvs[k].append(r2['cvs'][0])
                    ents[k].append(r2['ents'][0])
                    isi_5ps[k].append(r2['isi_5p'][0])

                result[k].update({'inputs': inputs,
                                  'corrected_rates': corrected_rates[k],
                                  'mean_rates': mean_rates[k],
                                  'cvs': cvs[k],
                                  'ents': ents[k],
                                  'isi_5p': isi_5ps[k],
                                  'io_properties': io_curve_stats(inputs, corrected_rates[k], label=k+'-rate',
                                                                  unit='Hz', verbose=True)})
                if plot:
                    res = [corrected_rates[k], mean_rates[k], cvs[k], ents[k], isi_5ps[k]]
                    labels = [r"$\nu^{\mathrm{corr}}_{\mathrm{out}}$", r"$\nu_{\mathrm{out}}$",
                              r"$\mathrm{CV_{ISI}}$", r"$\mathrm{H_{ISI}}$", r"$\mathrm{ISI_{5p}}$"]
                    colors = viz.helper.get_cmap(len(res), cmap='viridis')

                    fig, axes = plt.subplots(1, len(res), figsize=(20, 5))
                    fig.suptitle("Population {}".format(k))
                    for idx, (r, lab, ax) in enumerate(zip(res, labels, axes)):
                        if idx == 0:
                            inputs /= 1000.
                        if idx == 0 or idx == 1:
                            viz.plotting.plot_io_curve(inputs[:-1], inputs[:-1], ax=ax, save=False, display=False,
                                          **{'color': 'r', 'linestyle': '--', 'linewidth': 2.})
                        viz.plotting.plot_io_curve(inputs[:-1], r, ax=ax, save=False, display=False,
                                      **{'ylabel': lab, 'linewidth': 2., 'color': colors(idx)})
                        ax.set_title(lab)
                    if save:
                        viz.helper.fig_output(fig, display=display, save=store+'-transfer-functions.pdf')
                    else:
                        viz.helper.fig_output(fig, display=display, save=save)
        else:
            result = {}
        return result

    # def self_sustaining_lifetime(self, network, analysis_interval, input_off, verbose=True):
    #     store = self.storage_paths['figures'] + self.storage_paths['label']
    #
    #     if self.network_type == 'SpikingNetwork':
    #         r1 = met.ssa_lifetime(network, analysis_interval, input_off=input_off,
    #                               display=verbose)


    # def compile_results(self):
    #     """
    #     Returns the main data to be stored
    #     :return:
    #     """
    #     out_dict = {'network_type': self.network_type, 'network_label': self.network_label}
    #     return out_dict


# ######################################################################################################################
def decoder_stability(decoders, storage_paths, cmap="viridis", display=False, save=False):
    store = storage_paths['figures']+storage_paths['label']
    readout_labels = [x.label+'-'+x.task for x in decoders]
    r_colors = viz.helper.get_cmap(len(readout_labels), cmap)

    # Weight evolution
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for idx, x in enumerate(decoders):
        final_norm, training_norm = x.weights.measure_stability()
        ax.plot(training_norm, lw=2, color=r_colors(idx), label=readout_labels[idx])
    ax.set_xlabel("Training")
    ax.set_ylabel(r"$|\mathrm{W^{out}}|$")
    plt.legend()
    if save:
        viz.helper.fig_output(fig, display, save=store+"-wout-norm.pdf")
    else:
        viz.helper.fig_output(fig, display, save=store)


def decoder_confusion(decoders, storage_paths, display=False, save=False):
    store = storage_paths['figures']+storage_paths['label']

    if len(decoders) >= 4:
        fig, ax = plt.subplots(int(np.ceil(np.sqrt(len(decoders)))), int(np.ceil(np.sqrt(len(decoders)))))
    elif len(decoders) == 1:
        fig, ax = plt.subplots()
        ax = [ax]
    else:
        fig, ax = plt.subplots(1, len(decoders))

    for ii, dc in enumerate(decoders):
        if save:
            viz.plotting.plot_confusion_matrix(dc.performance['confusion_matrix'], ax=ax[ii],
                                               label=dc.task, ax_label=dc.label+'-'+dc.task,
                                               display=display, save=store+dc.label+'-'+dc.task+'-confusion.pdf')
        else:
            viz.plotting.plot_confusion_matrix(dc.performance['confusion_matrix'], ax=ax[ii],
                                               label=dc.task, ax_label=dc.label+'-'+dc.task,
                                               display=display, save=False)


def decoder_weights(decoders, storage_paths, display=False, save=False):
    store = storage_paths['figures']+storage_paths['label']
    for r0 in decoders:
        if save:
            r0.weights.plot_weights(display=display, save=store + '-{}-{}-weights'.format(r0.label, r0.task))
        else:
            r0.weights.plot_weights(display=display, save=False)


def decoder_evolution(decoders, storage_paths, parameters, display=False, save=False):
    store = storage_paths['figures']+storage_paths['label']
    for r0 in decoders:
        # look only at last batch in each epoch
        labels = ["train_batch={}_epoch={}".format(parameters.task_pars.n_batches, n_ep+1) for n_ep in range(parameters.task_pars.n_epochs)]

        if not empty(r0.weights.batch_weights):
            plot_grid = int(np.ceil(np.sqrt(len(labels))))
            fig, ax = plt.subplots(plot_grid, plot_grid, figsize=(20, 20))
            fig.suptitle("{}".format(r0.label+"-"+r0.task))

            axes = list(itertools.chain(*ax))
            data = []
            ax_lab = []

            for b_lab, w_out in zip(r0.weights.batch_labels, r0.weights.batch_weights):
                if b_lab in labels:
                    idx = labels.index(b_lab)
                    data.append(w_out)
                    ax_lab.append(b_lab.split('_')[-1])

            bins = [100 for _ in range(len(data))]

            for axx, lab in zip(axes, ax_lab):
                axx.set_title(lab)

            if save:
                viz.plotting.plot_histograms(axes[:len(data)], data, bins, cmap="viridis", kde=False, display=display,
                                             save=store+'-{}-{}-weight-evolution.pdf'.format(r0.label, r0.task))
            else:
                viz.plotting.plot_histograms(axes[:len(data)], data, bins, cmap="viridis", kde=False, display=display, save=False)