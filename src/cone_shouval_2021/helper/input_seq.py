"""

"""
import numpy as np

from fna.tasks.symbolic import ArtificialGrammar, SymbolicSequencer
from fna.tasks.symbolic.embeddings import VectorEmbeddings  # don't remove!
from fna.encoders import NESTEncoder, InputMapper
from fna.tools.utils import logger

import nest

logprint = logger.get_logger(__name__)

def _add_background_noise(snn, parameters):
    """
    Adds GWN background noise to each population.

    :param snn:
    :param parameters: mean and std given in parameters.input_pars.bg_noise_current
    :return:
    """
    if hasattr(parameters.input_pars, 'bg_noise_current'):
        print('Creating GWN background noise current generator with std', parameters.input_pars.bg_noise_current)
        noise_gen = nest.Create('noise_generator', 1, {'mean': parameters.input_pars.bg_noise_current[0],
                                                       'std': parameters.input_pars.bg_noise_current[1]})

        for pop_name, pop_obj in snn.populations.items():
            nest.Connect(noise_gen, pop_obj.gids)
    else:
        logprint.warning("No background noise defined! Missing 'bg_noise_current' parameter?")


def fna_init_input(parameters, snn):
    """
    Create stimulus sequence and compute reward times based on stimulus duration.

    :return:
    """
    sequencer_pars = parameters.input_pars.discrete_sequencer[1]
    n_strings = parameters.task_pars.batch_size
    input_pars = parameters.input_pars

    elman = ArtificialGrammar(**sequencer_pars)
    elman.generate_string_set(n_strings=n_strings, correct_strings=True)

    ######################################
    embedding_obj = globals()[input_pars.embedding[0]](vocabulary=elman.tokens)
    embedding = getattr(embedding_obj, input_pars.embedding[1])()
    embedded_signal = embedding.unfold(**input_pars.unfolding)

    # need to reset eos amplitude to 0
    eos_idx = embedded_signal.vocabulary.index('#')
    embedded_signal.stimulus_set['#'][0].analog_signals[eos_idx].signal.fill(0.)  # set amp for EOS symbol to 0

    # resort intervals from alphabet according to the order in the embedding vocabulary
    # duration of the silent period for each token
    intervals = [d - input_pars.unfolding.duration for d in input_pars.misc.token_duration]
    intervals = [intervals[sequencer_pars['alphabet'].index(e)] for e in embedding.vocabulary]

    # inhomogeneous Poisson generator
    # create it without a stim_seq initially, update state later on during each batch separately
    encoder = NESTEncoder('inhomogeneous_poisson_generator', label='poisson-input',
                          dim=embedding.embedding_dimensions, input_resolution=parameters.kernel_pars.resolution)

    # input synapses
    input_syn = {'model': 'static_synapse', 'delay': 1., 'weight': parameters.connection_pars.conns.w_in['weight'],
                 'receptor_type': 1}
    input_conn = {'rule': 'all_to_all'}

    input_synapses = {
        'connect_populations': [],
        'weight_matrix': [],
        'conn_specs': [],
        'syn_specs': []
    }

    for p in snn.population_names:
        if hasattr(input_pars.misc, 'stim_inh_pop') and input_pars.misc.stim_inh_pop and 'L5' not in p:
            continue
        elif not (hasattr(input_pars.misc, 'stim_inh_pop') and input_pars.misc.stim_inh_pop) \
                and ('L5' not in p or 'I' in p):
            continue

        wm = np.zeros((embedding.embedding_dimensions, snn.n_neurons[0]))
        token_idx = int(p.split('_')[-1]) + 1  # +1 because '#' is the first symbol
        wm[token_idx, :] = parameters.connection_pars.conns.w_in['weight']
        input_synapses['connect_populations'].append((p, encoder.name))
        input_synapses['weight_matrix'].append(wm)
        input_synapses['conn_specs'].append(input_conn)
        input_synapses['syn_specs'].append(input_syn)

    # connect input
    InputMapper(source=encoder, target=snn, parameters=input_synapses)

    _add_background_noise(snn, parameters)

    return encoder, elman, embedded_signal, intervals


def get_sample_sequence_from_batch(batch_data, sample_idx):
    batch_seq = batch_data['batch_seq']
    start_pos_seq = 0
    sample_cnt = 0

    for idx, t in enumerate(batch_seq):
        if t == '#':
            if idx < len(batch_seq) - 1 and sample_cnt < sample_idx:
                start_pos_seq = idx + 1
            sample_cnt += 1

    stop_pos_seq = start_pos_seq
    while batch_seq[stop_pos_seq] != '#':
        stop_pos_seq += 1

    return batch_seq[start_pos_seq:stop_pos_seq+1], batch_data['batch_token_intervals'][start_pos_seq:stop_pos_seq+1]

