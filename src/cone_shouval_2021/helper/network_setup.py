import logging
import sys
import copy
import numpy as np
from matplotlib import pyplot as plt

from . import nestml_loader
from . import parse_params
from . import is_etrace
from . import naming

from fna.tools.utils.system import reset_nest_kernel
from fna.tools.network_architect.connectivity import NESTConnector
from fna.networks.snn import SpikingNetwork
from fna.decoders.extractors import set_recording_device

import nest


def get_sparse_weight_matrix(N, weight_dict):
    """
    Returns a sparse weight matrix. Currently only the pairwise_bernoulli rule is supported.
    :param N:
    :param weight_dict:
    :return:
    """
    W = np.random.binomial(1, weight_dict['density'], (N, N)).astype(float) #TODO temp
    W *= weight_dict['weight']

    np.fill_diagonal(W, 0.)

    return W


def get_randomized_weight_matrix(N, weight_dict):
    """
    Computes a sparse, randomized weight matrix according to a Gaussian/uniform distribution.
    :return:
    """
    W = get_sparse_weight_matrix(N, weight_dict)
    W_res = W

    if hasattr(weight_dict, 'distribution') and weight_dict['distribution'] is not None:
        print("Randomizing (Gaussian) weight matrix with params: {}".format(weight_dict['distribution']))
        W_rand = weight_dict['distribution'][0](**weight_dict['distribution'][1], size=(N, N))
        W_bin = copy.deepcopy(W)  # bonary conn matrix
        W_bin[W_bin != 0.] = 1.
        W_res = abs(W + np.multiply(W_rand, W_bin))

        # for inhibitory weights, multiply with -1
        if weight_dict['weight'] < 0.:
            W_res *= -1

        def _plot_hist(W_):
            hist, bins = np.histogram(W_, bins=200)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            plt.bar(center, hist, align='center', width=width)
            plt.yscale('log')
            plt.show()

        print('Nonzero entries in spares T->T matrix: {}'.format(np.count_nonzero(W)))

    return W_res


def connect_populations(parameters, syn_spec_plastic_rec, syn_spec_plastic_ff, syn_spec_static):
    """
    Create dictionary used by FNA for the connection between all populations.
    Connections between intracolumnar T->T and cross-columnar M->T populations are fixed (non-randomized), but
    all other connections have a Gaussian distribution with std 0.1.

    Connections E->E and E->I have a synaptic time constant of 80 ms, whereas I->E have 10 ms.

    :param parameters:
    :param syn_spec_plastic_rec
    :param syn_spec_plastic_ff
    :param syn_spec_static
    :return:
    """
    conn_pars = parameters.connection_pars
    populations = parameters.net_pars.populations

    snn_synapses = {
        'connect_populations': [],
        'weight_matrix': [],
        'conn_specs': [],
        'syn_specs': []
    }
    weight_var = 'weight' if is_etrace(parameters) or parameters.connection_pars.plastic_syn_model is None else 'w'
    n_cols = len(parameters.net_pars.populations) // 4
    # conn_rule = {}
    conn_rule = {'rule': 'pairwise_bernoulli', 'p': 0.26, 'autapses': False}

    for col_id in range(n_cols):
        # intracolumnar
        pop_T, pop_T_inh, pop_M, pop_M_inh = populations[col_id * 4 : (col_id + 1) * 4]
        population_sizes = parameters.net_pars.population_size[col_id * 4 : (col_id + 1) * 4]
        assert len(np.unique(population_sizes)) == 1, "Populations must have the same size!"
        N = population_sizes[0]

        # the order of these connections should not be changed!!! downstream code makes critical assumptions
        snn_synapses['connect_populations'].extend([
            (pop_T, pop_T),         # T->T
            (pop_M, pop_T),         # T->M
            (pop_T_inh, pop_T),     # T->I (L5)
            (pop_M_inh, pop_M),     # M->I (L23)

            # (pop_T, pop_T_inh),     # I (L5) -> T
            (pop_M, pop_T_inh),     # I (L5) -> M
        ])
        n_conn = 5

        # according to the connectivity order above
        syn_spec_list = [copy.deepcopy(syn_spec_plastic_rec[0])]
        syn_spec_list.extend([copy.deepcopy(syn_spec_static[0]) for _ in range(n_conn - 1)])

        # add timer syn spec and weight matrix
        syn_spec_list[0].update({weight_var: conn_pars.conns.w_EE_L5_rec['weight'],
                                 'weight': conn_pars.conns.w_EE_L5_rec['weight'],
                                 'delay': conn_pars.conns.w_EE_L5_rec['delay'],
                                 'receptor_type': 0})        # T -> T
        W_timer_rec = get_sparse_weight_matrix(population_sizes[0], conn_pars.conns.w_EE_L5_rec)
        snn_synapses['weight_matrix'].append(W_timer_rec)
        snn_synapses['conn_specs'].append(conn_rule)  # add empty dictionary for recurrent connections

        # add info for all other connections
        nontimer_weight_list = [None, conn_pars.conns.w_EE_L23_L5,
                                conn_pars.conns.w_IE_L5_rec,
                                conn_pars.conns.w_IE_L23_rec,
                                # conn_pars.conns.w_EI_L5_rec,
                                conn_pars.conns.w_EI_L23_L5,
                                ]  # order is critical!
        for idx in range(1, n_conn):
            rec_type = 0 if idx <= 3 else 1  # connections from E populations have 80 ms, from I only 10 ms.
            syn_spec_list[idx].update({'weight': nontimer_weight_list[idx]['weight'],
                                       'delay': nontimer_weight_list[idx]['delay'],
                                       'receptor_type': rec_type})

            tmp_W = get_randomized_weight_matrix(N, nontimer_weight_list[idx])

            snn_synapses['weight_matrix'].append(tmp_W)
            snn_synapses['conn_specs'].append(conn_rule)

        snn_synapses['syn_specs'].extend(syn_spec_list)  # gathered in a list before, now update syn_specs

        # intercolumnar
        for col_id_cross in range(n_cols):
            if col_id_cross == col_id:  # avoid intracolumnar M->T connections
                continue

            pop_T_cross = naming.get_pop_name('timer_exc', col_id_cross)
            pop_M_cross = naming.get_pop_name('messenger_exc', col_id_cross)

            snn_synapses['connect_populations'].extend([
                (pop_T_cross, pop_M),         # M -> T_cross

                (pop_T_cross, pop_T_inh),     # I (L5) -> T_cross
                (pop_M_cross, pop_M_inh),     # I (L23) -> M_cross
            ])

            n_conn = 3

            if not conn_pars.ff_all:
                # only create feed-forward (sequential) intercolumnar excitatory connections.
                # A matrix of 0s ensures this because when a W matrix is given, FNA doesn't create additional conns
                if col_id_cross != col_id + 1:
                    tmp_ff_W = np.zeros((N, N))
                else:
                    # setting None here ensures that sequential FF connection are connected via the bernoulli rule
                    tmp_ff_W = None
            else:
                tmp_ff_W = None  # None here ensures all-to-all FF conns, cause the pairwise bernoulli will create them

            # according to the connectivity order above
            # M->T cross-columnar excitatory connections
            syn_spec_list = [copy.deepcopy(syn_spec_plastic_ff[0])]  # use the FF dictionary, may be different
            syn_spec_list[0].update({weight_var: conn_pars.conns.w_EE_L5_L23_cross['weight'],
                                     'weight': conn_pars.conns.w_EE_L5_L23_cross['weight'],
                                     'delay': conn_pars.conns.w_EE_L5_L23_cross['delay'],
                                     'receptor_type': 0})
            snn_synapses['weight_matrix'].append(tmp_ff_W)

            # I(M)->M and I(T)->T cross-columnar inhibitory connections
            syn_spec_list.extend([copy.deepcopy(syn_spec_static[0]) for _ in range(n_conn - 1)])
            syn_spec_list[1].update({'weight': conn_pars.conns.w_EI_L5_L5_cross['weight'],
                                     'delay': conn_pars.conns.w_EI_L5_L5_cross['delay'],
                                     'receptor_type': 1})
            syn_spec_list[2].update({'weight': conn_pars.conns.w_EI_L23_L23_cross['weight'],
                                     'delay': conn_pars.conns.w_EI_L23_L23_cross['delay'],
                                     'receptor_type': 1})

            # I(M)->M
            tmp_W = get_randomized_weight_matrix(N, conn_pars.conns.w_EI_L5_L5_cross)
            snn_synapses['weight_matrix'].append(tmp_W)

            # I(T)->T
            tmp_W = get_randomized_weight_matrix(N, conn_pars.conns.w_EI_L23_L23_cross)
            snn_synapses['weight_matrix'].append(tmp_W)

            snn_synapses['conn_specs'].extend([conn_rule] * n_conn)
            snn_synapses['syn_specs'].extend(syn_spec_list)

    # need to manually name the synapses here for FNA, otherwise it's not possible to create multiple types of
    # connections between the same populations.
    snn_synapses['synapse_names'] = []
    for idx in range(len(snn_synapses['syn_specs'])):
        syn_name = '{}_{}_{}'.format(snn_synapses['connect_populations'][idx][1], # TODO swap these to 0, 1
                                     snn_synapses['connect_populations'][idx][0],
                                     snn_synapses['syn_specs'][idx]['model'])
        snn_synapses['synapse_names'].append(syn_name)

    return snn_synapses


def connect_populations_m2(parameters, syn_spec_plastic_rec, syn_spec_plastic_ff, syn_spec_static):
    """
    Create dictionary used by FNA for the connection between all populations.
    Connections between intracolumnar T->T and cross-columnar M->T populations are fixed (non-randomized), but
    all other connections have a Gaussian distribution with std 0.1.

    Connections E->E and E->I have a synaptic time constant of 80 ms, whereas I->E have 10 ms.

    :param parameters:
    :param syn_spec_plastic_rec
    :param syn_spec_plastic_ff
    :param syn_spec_static
    :return:
    """
    conn_pars = parameters.connection_pars
    populations = parameters.net_pars.populations

    snn_synapses = {
        'connect_populations': [],
        'weight_matrix': [],
        'conn_specs': [],
        'syn_specs': []
    }
    weight_var = 'weight' if is_etrace(parameters) or parameters.connection_pars.plastic_syn_model is None else 'w'
    n_cols = len(parameters.net_pars.populations) // 4
    conn_rule = {'rule': 'pairwise_bernoulli', 'p': 0.26, 'autapses': False}

    for col_id in range(n_cols):
        # intracolumnar
        pop_T, pop_T_inh, pop_M, pop_M_inh = populations[col_id * 4 : (col_id + 1) * 4]
        population_sizes = parameters.net_pars.population_size[col_id * 4 : (col_id + 1) * 4]
        assert len(np.unique(population_sizes)) == 1, "Populations must have the same size!"
        N = population_sizes[0]

        # the order of these connections should not be changed!!! downstream code makes critical assumptions
        snn_synapses['connect_populations'].extend([
            (pop_T, pop_T),         # T->T
            (pop_M, pop_T),         # T->M
            (pop_M_inh, pop_T),     # T->I (L23)

            (pop_T, pop_T_inh),     # I (L5) -> T
            (pop_M, pop_M_inh),     # I (L23) -> M
        ])
        n_conn = 5

        # according to the connectivity order above
        syn_spec_list = [copy.deepcopy(syn_spec_plastic_rec[0])]
        syn_spec_list.extend([copy.deepcopy(syn_spec_static[0]) for _ in range(n_conn - 1)])

        # add timer syn spec and weight matrix
        syn_spec_list[0].update({weight_var: conn_pars.conns.w_EE_L5_rec['weight'],
                                 'weight': conn_pars.conns.w_EE_L5_rec['weight'],
                                 'delay': conn_pars.conns.w_EE_L5_rec['delay'],
                                 'receptor_type': 0})        # T -> T
        W_timer_rec = get_sparse_weight_matrix(population_sizes[0], conn_pars.conns.w_EE_L5_rec)
        snn_synapses['weight_matrix'].append(W_timer_rec)
        snn_synapses['conn_specs'].append(conn_rule)  # add empty dictionary for recurrent connections

        # add info for all other connections
        nontimer_weight_list = [
            None, conn_pars.conns.w_EE_L23_L5,  # T -> M
            conn_pars.conns.w_IE_L23_L5,        # T -> I (L23)

            conn_pars.conns.w_EI_L5_L5,
            conn_pars.conns.w_EI_L23_L23,
        ]  # order is critical!

        for idx in range(1, n_conn):
            rec_type = 0 if idx <= 2 else 1  # connections from E populations have 80 ms, from I only 10 ms.
            syn_spec_list[idx].update({'weight': nontimer_weight_list[idx]['weight'],
                                       'delay': nontimer_weight_list[idx]['delay'],
                                       'receptor_type': rec_type})

            tmp_W = get_randomized_weight_matrix(N, nontimer_weight_list[idx])

            snn_synapses['weight_matrix'].append(tmp_W)
            snn_synapses['conn_specs'].append(conn_rule)

        snn_synapses['syn_specs'].extend(syn_spec_list)  # gathered in a list before, now update syn_specs

        # intercolumnar
        for col_id_cross in range(n_cols):
            if col_id_cross == col_id:  # avoid intracolumnar M->T connections
                continue

            pop_T_cross = naming.get_pop_name('timer_exc', col_id_cross)
            pop_T_cross_inh = naming.get_pop_name('timer_inh', col_id_cross)
            pop_M_cross = naming.get_pop_name('messenger_exc', col_id_cross)
            pop_M_cross_inh = naming.get_pop_name('messenger_inh', col_id_cross)

            snn_synapses['connect_populations'].extend([
                (pop_T_cross, pop_M),         # M -> T_cross
                (pop_T_cross_inh, pop_T),     # I (L5) -> T_cross
                (pop_M_cross_inh, pop_M),     # I (L23) -> M_cross
            ])

            n_conn = 3
            if not conn_pars.ff_all:
                # only create feed-forward (sequential) intercolumnar excitatory connections.
                # A matrix of 0s ensures this because when a W matrix is given, FNA doesn't create additional conns
                if col_id_cross != col_id + 1:
                    tmp_ff_W = np.zeros((N, N))
                else:
                    # tmp_ff_W = get_randomized_weight_matrix(N, conn_pars.conns.w_EE_L5_L23_cross)
                    # setting None here ensures that sequential FF connection are connected via the bernoulli rule
                    tmp_ff_W = None
            else:
                tmp_ff_W = None  # None here ensures all-to-all FF conns, cause the pairwise bernoulli will create them

            # according to the connectivity order above
            # M->T cross-columnar excitatory connections
            syn_spec_list = [copy.deepcopy(syn_spec_plastic_ff[0])]  # use the FF dictionary, may be different
            syn_spec_list[0].update({weight_var: conn_pars.conns.w_EE_L5_L23_cross['weight'],
                                     'weight': conn_pars.conns.w_EE_L5_L23_cross['weight'],
                                     'delay': conn_pars.conns.w_EE_L5_L23_cross['delay'],
                                     'receptor_type': 0})
            snn_synapses['weight_matrix'].append(tmp_ff_W)

            # I(M)->M and I(T)->T cross-columnar inhibitory connections
            syn_spec_list.extend([copy.deepcopy(syn_spec_static[0]) for _ in range(n_conn - 1)])
            syn_spec_list[1].update({'weight': conn_pars.conns.w_IE_L5_L5_cross['weight'],
                                     'delay': conn_pars.conns.w_IE_L5_L5_cross['delay'],
                                     'receptor_type': 0})
            syn_spec_list[2].update({'weight': conn_pars.conns.w_IE_L23_L23_cross['weight'],
                                     'delay': conn_pars.conns.w_IE_L23_L23_cross['delay'],
                                     'receptor_type': 0})

            # I(M)->M
            tmp_W = get_randomized_weight_matrix(N, conn_pars.conns.w_IE_L5_L5_cross)
            snn_synapses['weight_matrix'].append(tmp_W)

            # I(T)->T
            tmp_W = get_randomized_weight_matrix(N, conn_pars.conns.w_IE_L23_L23_cross)
            snn_synapses['weight_matrix'].append(tmp_W)

            snn_synapses['conn_specs'].extend([conn_rule] * n_conn)
            snn_synapses['syn_specs'].extend(syn_spec_list)

    # need to manually name the synapses here for FNA, otherwise it's not possible to create multiple types of
    # connections between the same populations.
    snn_synapses['synapse_names'] = []
    for idx in range(len(snn_synapses['syn_specs'])):
        syn_name = '{}_{}_{}'.format(snn_synapses['connect_populations'][idx][1], # TODO swap these to 0, 1
                                     snn_synapses['connect_populations'][idx][0],
                                     snn_synapses['syn_specs'][idx]['model'])
        snn_synapses['synapse_names'].append(syn_name)

    return snn_synapses


def _add_recorders(parameters, is_etrace=False):
    """
    Returns the recording devices (spikes and voltmeter) for each population, as per the recording parameters.
    TODO implement per population handling
    :param parameters:
    :param is_etrace:
    :return:
    """
    snn_pars = parameters.net_pars
    rec_pars = parameters.recording_pars
    resolution = parameters.kernel_pars.resolution

    # Attach measurement devices (spike recorder and voltmeter).
    spike_recorder = set_recording_device(start=0.0,
                                          stop=sys.float_info.max,
                                          resolution=resolution,
                                          record_to='memory',
                                          device_type='spike_detector')
    voltmeter = set_recording_device(start=0.0,
                                     stop=sys.float_info.max,
                                     resolution=resolution,
                                     record_to='memory',
                                     device_type='multimeter')
    if is_etrace:
        voltmeter.update({'record_from': ['V_m', 'rates'], 'record_n': None})
    else:
        voltmeter.update({'record_from': ['V_m'], 'record_n': None})

    spike_recorders = []
    voltmeters = []

    for pop_name in snn_pars['populations']:
        if 'record_spikes_pop' in rec_pars: #and pop_name in rec_pars['record_spikes_pop']:
            spike_recorders.append(spike_recorder)
        else:
            spike_recorders.append(None)

        if 'record_vm_pop' in rec_pars and pop_name in rec_pars['record_vm_pop']:
            voltmeters.append(voltmeter)
        else:
            voltmeters.append(None)

    spike_recorders = None if 'record_spikes_pop' not in rec_pars else spike_recorders
    voltmeters = None

    return spike_recorders, voltmeters


def build_network(parameters, input_dimensions=None, output_dimensions=None, signal=None, batch_time=None,
                  symbolic=False, batch_data=False):
    """
    Setup network model and retrieve all relevant network-specific parameters
    :param parameters: Experiment ParameterSet
    :param input_dimensions: int
    :param output_dimensions: int
    :param signal:
    :param batch_time:
    :param symbolic: bool
    :param batch_data: bool - whether to retrieve system-specific information for batch preparation
    :return:
    """
    nestml_loader.install_dynamic_modules('ShouvalNeuron')
    nestml_loader.install_dynamic_modules('shouval_multith_synapse')
    reset_nest_kernel(parameters.kernel_pars)

    # synapses and connections
    syn_model = parameters.connection_pars.plastic_syn_model
    syn_spec_plastic_rec, syn_spec_plastic_ff, syn_spec_static = \
        parse_params.get_synaptic_properties(parameters, syn_model, [])

    if 'model2' not in parameters.connection_pars or parameters.connection_pars.model2 is False:
        snn_synapses = connect_populations(parameters, syn_spec_plastic_rec, syn_spec_plastic_ff, syn_spec_static)
    else:
        snn_synapses = connect_populations_m2(parameters, syn_spec_plastic_rec, syn_spec_plastic_ff, syn_spec_static)

    # get recording devices
    spike_recorders, voltmeters = _add_recorders(parameters, is_etrace=False)

    snn = SpikingNetwork(parameters.net_pars, spike_recorders=spike_recorders, analog_recorders=voltmeters,
                         label=parameters.label)
    NESTConnector(source_network=snn, target_network=snn, connection_parameters=snn_synapses)

    return snn
