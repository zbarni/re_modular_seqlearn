from tqdm import tqdm
import numpy as np
import types
import itertools

from fna.tasks.symbolic.embeddings import VectorEmbeddings
from fna.tasks.preprocessing import ImageFrontend
from fna.tasks.analog import narma


# ######################################################################################################################
def prepare_symbolic_batch(simulator, n_batches, batch_size, sequencer, discrete_embedding=None,
                           continuous_embedding=None, batch_time=None, as_tensor=False, signal_pars=None, noise=None,
                           decoder_output_pars=None):
    """
    Generates data batches for symbolic processing tasks
    :param n_batches: Number of unique batches
    :param batch_size: Size of batch (discrete steps or time)
    """
    if simulator == 'Tensorflow' or simulator == 'TensorFlow' or simulator == 'TF' or simulator == 'Python':
        if not as_tensor:
            inputs = []
            targets = []
        else:
            inputs = np.empty(shape=(batch_time, int(n_batches), int(continuous_embedding.embedding_dimensions)),
                              dtype=np.float32)
            targets = np.empty(shape=(batch_time, int(n_batches), int(len(sequencer.tokens))), dtype=np.float32)

        decoder_targets = []
        for batch in tqdm(range(n_batches), desc="Generating batches"):
            # input sequence
            batch_seq = sequencer.generate_random_sequence(T=batch_size, verbose=False)  # TODO modify (provide function)!!
            input_times = np.arange(len(batch_seq))

            # decoder outputs
            if decoder_output_pars is not None:
                batch_targets = sequencer.generate_default_outputs(batch_seq, verbose=False, **decoder_output_pars)
                decoder_targets.append(batch_targets)

            # network inputs
            if discrete_embedding is not None and not as_tensor:
                input_batch = discrete_embedding.draw_stimulus_sequence(batch_seq, as_array=True, verbose=False)
                inputs.append(input_batch.T)

            if continuous_embedding is not None and as_tensor:
                if isinstance(continuous_embedding, ImageFrontend):
                    input_batch, input_times = continuous_embedding.draw_stimulus_sequence(
                        batch_seq, continuous=True, unfold=True, onset_time=0., verbose=False)
                else:
                    input_batch, input_times = continuous_embedding.draw_stimulus_sequence(
                        batch_seq, continuous=True, onset_time=0., verbose=False)

                if isinstance(input_batch, types.GeneratorType):
                    for t, inp in enumerate(input_batch):
                        inputs[int(inp[1][0]/inp[0].dt):int(inp[1][1]/inp[0].dt), batch, :] = inp[0].as_array().T
                else:
                    inputs[:, batch, :] = input_batch.as_array().astype(np.float32).T

            # network targets
            target_seq = sequencer.generate_default_outputs(batch_seq, max_memory=0, max_chunk=0, max_prediction=0)[0][
                'output']  # classification target for main network output (in ANNs)
            if discrete_embedding is not None and continuous_embedding is None:
                target_batch = VectorEmbeddings(vocabulary=sequencer.tokens).one_hot().draw_stimulus_sequence(
                    target_seq, as_array=True, verbose=False)
                targets.append(target_batch.T)
            if continuous_embedding is not None:
                output_signal_pars = (lambda a, b: a.update(b) or a)(signal_pars, {'kernel': ('box', {}),
                                                                                   'amplitude': 1.})
                # out_signal = VectorEmbeddings(vocabulary=np.unique(target_seq)).one_hot().unfold(verbose=False,
                #                                                                                  **output_signal_pars)
                out_signal = VectorEmbeddings(vocabulary=continuous_embedding.vocabulary).one_hot()
                out_signal = out_signal.unfold(verbose=False, **output_signal_pars)
                target_batch = out_signal.draw_stimulus_sequence(target_seq, continuous=True, verbose=False)
                targets[:, batch, :] = target_batch[0].as_array().astype(np.float32).T

        # add noise to inputs
        if noise:
            input_noise = np.random.normal(noise['mean'], noise['std'], inputs.shape)
            inputs += input_noise
            inputs[inputs < 0.] = 0.  # rectify

        return {'inputs': inputs, 'targets': targets, 'decoder_outputs': decoder_targets, 'input_times': input_times}

    elif simulator == 'NEST':
        batch_sequence = [sequencer.generate_random_sequence(T=batch_size, verbose=False) for _ in range(n_batches)]
        batch_targets = [sequencer.generate_default_outputs(batch_seq, verbose=False, **decoder_output_pars)
                         for batch_seq in batch_sequence]

        return {'inputs': batch_sequence, 'decoder_outputs': batch_targets}


# ######################################################################################################################
def prepare_analog_batch(simulator, n_batches, batch_size, sequencer, discrete_embedding=None,
                         continuous_embedding=None, batch_time=None, as_tensor=False, signal_pars=None,
                         decoder_output_pars=None):
    """
    Generates data batches for analog processing tasks
    :param n_batches: Number of unique batches
    :param batch_size: Size of batch (discrete steps or time)
    """
    if simulator == 'Tensorflow' or simulator == 'TensorFlow' or simulator == 'TF' or simulator == 'Python':
        decoder_targets = []
        if not as_tensor:
            inputs = []
            targets = []
        else:
            inputs = np.empty(shape=(batch_time, int(n_batches), int(1)),
                              dtype=np.float32)
            targets = np.empty(shape=(batch_time, int(n_batches), int(1)), dtype=np.float32)

        for batch in tqdm(range(n_batches), desc="Generating batches"):
            # inputs
            batch_seq = sequencer.generate_random_sequence(T=batch_size, verbose=False)
            if discrete_embedding is not None and continuous_embedding is None:
                signal = discrete_embedding.draw_stimulus_sequence(batch_seq, as_array=True, verbose=False)
                input_times = np.arange(len(batch_seq))
            if continuous_embedding is not None:
                signal, input_times = continuous_embedding.draw_stimulus_sequence(batch_seq, onset_time=0., continuous=True,
                                                                                  intervals=None, verbose=False)
                signal = signal.as_array()
            input_batch = signal #np.reshape(signal, (1, signal.shape[0]))

            # network target
            target_batch = input_batch

            # decoder outputs
            if signal_pars is not None:
                duration = signal_pars['duration']
            else:
                duration = None
            decoder_targets.append(generate_analog_targets(target_batch, sequencer, continuous_embedding, signal,
                                                      duration, decoder_output_pars))
            if not as_tensor:
                inputs.append(input_batch.T)
                targets.append(target_batch.T)
            else:
                inputs[:, batch, :] = input_batch.astype(np.float32).T
                targets[:, batch, :] = target_batch.astype(np.float32).T

        return {'inputs': inputs, 'targets': targets, 'decoder_outputs': decoder_targets, 'input_times': input_times}

    elif simulator == 'NEST':
        decoder_targets = []
        batch_sequence = []
        for batch in tqdm(range(n_batches), desc="Generating batches"):
            # inputs
            batch_seq = sequencer.generate_random_sequence(T=batch_size, verbose=False)
            batch_sequence.append(batch_seq)
            if discrete_embedding is not None and continuous_embedding is None:
                signal = discrete_embedding.draw_stimulus_sequence(batch_seq, as_array=True, verbose=False)
                input_times = np.arange(len(batch_seq))
            if continuous_embedding is not None:
                signal, input_times = continuous_embedding.draw_stimulus_sequence(batch_seq, onset_time=0., continuous=True,
                                                                                  intervals=None, verbose=False)
                signal = signal.as_array()
            input_batch = signal #np.reshape(signal, (1, signal.shape[0]))

            # network target (not applicable, but has to be present)
            target_batch = input_batch

            decoder_targets.append(generate_analog_targets(target_batch, sequencer, continuous_embedding, batch_seq,# signal,
                                                      signal_pars['duration'], decoder_output_pars))

        return {'inputs': batch_sequence, 'decoder_outputs': decoder_targets, 'input_times': input_times}


def generate_analog_targets(target_batch, sequencer, continuous_embedding, input_signal, stim_duration,
                            decoder_output_pars):
    batch_target = []
    if continuous_embedding is not None and stim_duration % 1. == 0.:
        dec_target = target_batch.ravel()[::int(stim_duration)]
    else:
        dec_target = target_batch.ravel()
    dec_target /= (dec_target.max() - dec_target.min())
    batch_target.append({
        'label': 'reconstruction',
        'output': dec_target,
        'accept': [True for _ in range(dec_target.shape[0])]})
    for k, v in decoder_output_pars.items():
        if k == 'narma':
            if isinstance(v, list):
                for order in v:
                    tgt = narma(dec_target, n=order)
                    acc = np.array([True for _ in range(dec_target.shape[0])])
                    # acc[np.where(tgt == 0.)] = False
                    # tgt[np.where(tgt == 0.)] = None

                    batch_target.append({
                        'label': '{}-{}'.format(k, order),
                        'output': list(tgt),
                        'accept': list(acc)})
            else:
                order = v
                tgt = narma(dec_target, n=order)
                acc = np.array([True for _ in range(dec_target.shape[0])])
                # acc[np.where(tgt == 0.)] = False
                # tgt[np.where(tgt == 0.)] = None

                batch_target.append({
                    'label': '{}-{}'.format(k, order),
                    'output': list(tgt),
                    'accept': list(acc)})
        elif k == 'memory':
            if not isinstance(input_signal, list):
                mem_tgts = sequencer.generate_default_outputs(list(input_signal[0, :]), verbose=False, max_memory=v)
            else:
                mem_tgts = sequencer.generate_default_outputs(input_signal, verbose=False, max_memory=v)
            for tgt in mem_tgts:
                batch_target.append(tgt)
    return batch_target


# def load_batch_from_array(inputs, simulator, n_batches, batch_size, as_tensor=False):
#
#     assert isinstance(inputs, list) and len(inputs) == n_batches, "Inputs should be a list of tuples (x, t), " \
#                                                                   "containing the input for each batch"
#
#     if simulator == 'Tensorflow' or simulator == 'TensorFlow' or simulator == 'TF' or simulator == 'Python':
#         decoder_targets = []
#         if not as_tensor:
#             inputs = []
#             targets = []
#         else:
#             inputs = np.empty(shape=(batch_size, int(n_batches), int(1)),
#                               dtype=np.float32)
#             targets = np.empty(shape=(batch_size, int(n_batches), int(1)), dtype=np.float32)
#
#         for batch in tqdm(range(n_batches), desc="Generating batches"):
#             signal, input_times = inputs[batch]

def load_profiler_batch(input_type, simulator, n_batches, batch_size, sequencer, discrete_embedding=None,
                        continuous_embedding=None, as_tensor=False, signal_pars=None,
                        decoder_output_pars={}, batch_params={}):
    if input_type in ["uniform", "uniform-noise", "gaussian", "gaussian-noise", "static"]:
        data_batch = prepare_analog_batch(simulator=simulator, n_batches=n_batches, batch_size=batch_size,
                                          sequencer=sequencer, discrete_embedding=discrete_embedding,
                                          decoder_output_pars=decoder_output_pars, **batch_params)
        return data_batch
    elif input_type == 'OU' or input_type == 'OU-noise':
        data_batch = []

    elif input_type in ['pulse', 'ramping']:
        if "continuous_embedding" in batch_params.keys():
            continuous_embedding = batch_params['continuous_embedding']
        if "signal_pars" in batch_params.keys():
            signal_pars = batch_params['signal_pars']
        if "as_tensor" in batch_params.keys():
            as_tensor = batch_params['as_tensor']

        if simulator == 'Tensorflow' or simulator == 'TensorFlow' or simulator == 'TF' or simulator == 'Python':
            decoder_targets = []
            if not as_tensor:
                inputs = []
                targets = []
            else:
                inputs = np.empty(shape=(batch_params['batch_time'], int(n_batches), int(1)),
                                  dtype=np.float32)
                targets = np.empty(shape=(batch_params['batch_time'], int(n_batches), int(1)), dtype=np.float32)

            for batch in tqdm(range(n_batches), desc="Generating batches"):
                # inputs
                if input_type == 'pulse':
                    batch_seq = list(itertools.chain(*['0', (batch_size-1)*'1'.split()]))
                elif input_type == 'ramping':
                    batch_seq = list(discrete_embedding.stimulus_set.keys())

                if discrete_embedding is not None and continuous_embedding is None:
                    signal = discrete_embedding.draw_stimulus_sequence(batch_seq, as_array=True, verbose=False)
                    input_times = np.arange(len(batch_seq))
                if continuous_embedding is not None:
                    signal, input_times = continuous_embedding.draw_stimulus_sequence(batch_seq, onset_time=0., continuous=True,
                                                                                      intervals=None, verbose=False)
                    signal = signal.as_array()
                input_batch = signal #np.reshape(signal, (1, signal.shape[0]))

                # network target
                target_batch = input_batch

                # decoder outputs
                if signal_pars is not None:
                    duration = signal_pars['duration']
                else:
                    duration = None
                decoder_targets.append(generate_analog_targets(target_batch, sequencer, continuous_embedding, signal,
                                                               duration, decoder_output_pars))
                if not as_tensor:
                    inputs.append(input_batch.T)
                    targets.append(target_batch.T)
                else:
                    inputs[:, batch, :] = input_batch.astype(np.float32).T
                    targets[:, batch, :] = target_batch.astype(np.float32).T

            return {'inputs': inputs, 'targets': targets, 'decoder_outputs': decoder_targets, 'input_times': input_times}

        elif simulator == 'NEST':
            decoder_targets = []
            batch_sequence = []
            for batch in tqdm(range(n_batches), desc="Generating batches"):
                # inputs
                if input_type == 'pulse':
                    batch_seq = list(itertools.chain(*['0', (batch_size-1)*'1'.split()]))
                elif input_type == 'ramping':
                    batch_seq = list(discrete_embedding.stimulus_set.keys())
                batch_sequence.append(batch_seq)
                if discrete_embedding is not None and continuous_embedding is None:
                    signal = discrete_embedding.draw_stimulus_sequence(batch_seq, as_array=True, verbose=False)
                    input_times = np.arange(len(batch_seq))
                if continuous_embedding is not None:
                    signal, input_times = continuous_embedding.draw_stimulus_sequence(batch_seq, onset_time=0., continuous=True,
                                                                                      intervals=None, verbose=False)
                    signal = signal.as_array()
                input_batch = signal #np.reshape(signal, (1, signal.shape[0]))

                # network target (not applicable, but has to be present)
                target_batch = input_batch

                decoder_targets.append(generate_analog_targets(target_batch, sequencer, continuous_embedding, batch_seq,# signal,
                                                               signal_pars['duration'], decoder_output_pars))

            return {'inputs': batch_sequence, 'decoder_outputs': decoder_targets, 'input_times': input_times}
    else:
        raise NotImplementedError("Batch generation for input type {} not currently supported".format(input_type))