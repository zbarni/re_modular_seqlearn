"""
Manual scaling by individual weight tuning.

Figure 5B.
"""

import copy
import itertools
import numpy as np
from defaults.paths import set_project_paths
from fna.tools.utils.system import set_kernel_defaults

from utils.system import set_system_parameters
from helper import naming

# ######################################################################################
# experiments parameters
project_label = 'cone_shouval_2021'
# experiment_label = 'test_rescale_set700_scale=EI_L5_L5_cross,IE_L5_rec,EE_L23_L5_short'
experiment_label = 'scale_manual'

# ######################################################################################
# system parameters
system_label = 'local'
paths = set_project_paths(system=system_label, project_label=project_label)

n_col = 3  # only use 3 columns for speed reasons
N = 400

# ################################
ParameterRange = {
    'p1': [0.02],
    'p2': [0.02],
    'T': np.arange(1, 3, 1)
}


# ################################
def build_parameters(p1, p2, T):
    system_params = set_system_parameters(cluster=system_label, nodes=1, ppn=6, mem=200000)

    # ############################################################
    # Simulation parameters
    resolution = 1.
    np_seed = 1
    kernel_pars = set_kernel_defaults(resolution=resolution, run_type=system_label, data_label=experiment_label,
                                      data_paths=paths, np_seed=np_seed, **system_params)

    # ###########################################################
    # Network parameters
    n_pop = 4 * n_col

    neuron_params_exc = dict(
        model='ShouvalNeuron',
        rho = 1 / 7.,
        gL = 10.,  # nS
        Cm = 20 * 10.,  # pF
        E_ex = -5.,
        E_L = -60.,
        E_in = -70.,
        # V_reset = -61.,
        V_reset = -60.,  # @critical - Matlab code uses a weird reset mechanism
        V_th = -55.,
        tau_syn_ex_rec0 = 80.,  # order critical: rec 0 = 80. !!!
        tau_syn_ex_rec1 = 10.,  # order critical: rec 1 = 10. !!!
        tau_syn_in = 10.,
        # tau_ref = 2.,
        t_ref = 3.,
        tau_w = 40.,  # firing rate filter time constant,
        V_m = -60.,
        I_e = 0.0,
    )
    neuron_params_inh = copy.deepcopy(neuron_params_exc)
    neuron_params_inh['V_th'] = -50.

    populations = []
    for col_id in range(n_col):
        populations.append(naming.get_pop_name('timer_exc', col_id))
        populations.append(naming.get_pop_name('timer_inh', col_id))
        populations.append(naming.get_pop_name('messenger_exc', col_id))
        populations.append(naming.get_pop_name('messenger_inh', col_id))

    net_params = {
        'type': 'SNN',
        'populations': populations,
        'population_size': [N for _ in range(n_pop)],
        'neurons': list(itertools.chain(*[[neuron_params_exc, neuron_params_inh, neuron_params_exc, neuron_params_inh]
                                          for _ in range(n_col)])),
        'randomize': [
            {}
            for _ in range(n_pop)
        ],
        'n_cols': n_col
    }

    delay = 1.
    density = 0.26
    wscale = 1. / np.sqrt(N/100)

    # naming convention is TGT_SRC, i.e., Target <- Source
    rescaled_weights = {
        'w_EE_L5_rec': 0.07 * wscale,
        'w_EE_L5_L23_cross': 0.0002 * wscale,
        'w_EE_L23_L5': 0.2 * wscale * 1.2,
        'w_EI_L23_L5': -70., #* wscale,
        'w_EI_L23_L23_cross': -100., #*wscale,
        # this must be scaled because it can create a dip in the FR of C_i+1, or worse, the next E_L5 won't fire
        # enough. Also, if FR I_L5 is too high, the M cells may fire too late and miss the Hebbian window / reward period
        'w_EI_L5_L5_cross': -100. * wscale * p1,
        'w_IE_L5_rec': 0.2 * wscale * p2,        # must be scaled because rate I_L5 is too high otherwise
        'w_IE_L23_rec': 1.,#*wscale,
    }
    learn_rate_ff = 0.25 * 1e3 / 12.5 /   10.  # for N = 400

    connection_params = {
        'conns': {
            'w_in': {
                'weight': 100.,
                'density': 1.,
                'distribution': None  # input weights are not randomized
            },

            # intracolumnar T->T
            'w_EE_L5_rec': {
                'weight': rescaled_weights['w_EE_L5_rec'],
                'density': density,
                'delay': delay,
                'distribution': None  # recurrent L5 weights are initially not randomized
            },

            # intracolumnar T->M
            'w_EE_L23_L5': {
                'weight': rescaled_weights['w_EE_L23_L5'],
                'density': density,
                'delay': delay,
                'distribution': None  # L5->L2/3 weights are initially not randomized
            },

            # cross-columnar M->T
            'w_EE_L5_L23_cross': {
                'weight': rescaled_weights['w_EE_L5_L23_cross'],
                'density': density,
                'delay': delay,
                'distribution': None
            },

            # intracolumnar I->E
            'w_EI_L5_rec': {  #
                'weight': 0.0,
                'density': density,
                'delay': delay,
                'distribution': (np.random.normal, {'loc': 0., 'scale': 0.1})
            },

            'w_EI_L23_L5': {
                'weight': rescaled_weights['w_EI_L23_L5'],
                'density': density,
                'delay': delay,
                'distribution': (np.random.normal, {'loc': 0., 'scale': 0.1})
            },

            # cross-columnar I->E
            'w_EI_L23_L23_cross': {
                'weight':rescaled_weights['w_EI_L23_L23_cross'],
                'density': density,
                'delay': delay,
                'distribution': (np.random.normal, {'loc': 0., 'scale': 0.1})
            },
            'w_EI_L5_L5_cross': {
                'weight': rescaled_weights['w_EI_L5_L5_cross'],
                'density': density,
                'delay': delay,
                'distribution': (np.random.normal, {'loc': 0., 'scale': 0.1})
            },

            # intracolumnar E->I
            'w_IE_L5_rec': {
                'weight': rescaled_weights['w_IE_L5_rec'],
                'density': density,
                'delay': delay,
                'distribution': (np.random.normal, {'loc': 0., 'scale': 0.1})
            },
            'w_IE_L23_rec': {
                'weight': rescaled_weights['w_IE_L23_rec'],
                'density': density,
                'delay': delay,
                'distribution': (np.random.normal, {'loc': 0., 'scale': 0.1})
            },
        },
        'ff_all': False,  # all-to-all FF connections between columns
        'plastic_syn_model': 'shouval_multith_synapse'
    }

    # ############################################################
    # Task / training parameters
    continuous = True  # yield each stimulus or draw each batch as a whole

    task_params = {
        'batch_size': 1,
        'batch_size_test': 1,
        'n_epochs': 1,
        'n_batches_transient': 1,
        'n_batches': 100,
        'n_batches_test': 10,
    }

    # ############################################################
    # Input parameters
    # discrete sequencer + embedding
    alphabet = ['A', 'B', 'C', 'D'][:n_col] + ['#']
    sequencer = ('ArtificialGrammar',
                 {
                     # Sequencer label
                     'label': 'Elman',

                     # list of unique tokens
                     'alphabet': alphabet,

                     # list of unique states in the graph (may or may not be = alphabet)
                     'states': alphabet,

                     # initial and terminal states
                     'start_states': ['A'],
                     'terminal_states': ['#'],

                     # state transition probabilities as a list of tuples (start, end, p)
                     'transitions': [(alphabet[i], alphabet[i + 1], 1.) for i in range(n_col)]
                 })
    embedding = ('VectorEmbeddings', 'one_hot', {})

    # continuous sequencers
    stim_duration = 50.
    stim_amplitude = 30.
    stim_kernel = ('box', {})

    signal_pars = {
        'to_signal': True,
        'duration': stim_duration,
        'amplitude': stim_amplitude,
        'kernel': stim_kernel,
        'dt': resolution,
    }

    input_params = {
        'discrete_sequencer': sequencer,
        'embedding': embedding,
        'continuous': continuous,
        'unfolding': signal_pars,
        'misc': {
            'token_duration': [700., 700., 700., 700.][:n_col] + [1000.],
            'trial_isi': 200.,
            'stim_inh_pop': True  # whether to stimulate the inhibitory populations in L5
        },
        'bg_noise_current': (0., 100. * wscale)
    }

    learning_params = {
        'tau_w': neuron_params_exc['tau_w'],
        'Tp_max_rec': 0.0033,
        'Td_max_rec': 0.00345,

        'Tp_max_ff': 0.0034,
        'Td_max_ff': 0.00345,

        'tau_ltp_rec': 2000.,
        'tau_ltd_rec': 1000.,
        'tau_ltp_ff': 200.,
        'tau_ltd_ff': 800.,

        'eta_ltp_rec': 45 * 3500.,
        'eta_ltd_rec': 25 * 3500.,
        'eta_ltp_ff': 20 * 3500.,
        'eta_ltd_ff': 15 * 3500.,

        'rate_th_rec': 0.01,
        'rate_th_ff': 0.02,

        'learn_rate_rec': 0.002 * 1e3 / 12.5,
        'learn_rate_ff': learn_rate_ff,

        'T_reward': 25.,
        'T_ref': 25.,
        'reward_delay': 25.,
        'weight_update_once_rec': 1,
        'weight_update_once_ff': 1,
        'norm_traces_rec': 0,  # whether to normalize traces as per the paper but not the code
        'norm_traces_ff': 0,
        'weights_ns': 1,  # whether weight unit is nS (0 means microS)
    }

    # ######################################################
    # decoding parameters
    decoding_parameters = {}
    extractor_parameters = {}

    decoder_outputs = {}

    n_samples_to_record = min(50, task_params['n_batches'])
    recording_params = {
        'sampling_interval': 100.,
        'n_syn_rec_mean': 500,
        'n_samples_to_record': n_samples_to_record,
        'samples_to_record': {
            'transient_epoch': {
                'transient_batch': [0]
            },
            '0': {
                str(x): [0] for x in np.linspace(0, task_params['n_batches'] - 1, n_samples_to_record).astype(int)
            },
            'test_epoch': {
                f'test_batch_{x}': [0] for x in range(task_params['n_batches_test'])
            }
        },
        'record_pop': [naming.get_pop_name(p, col_id)
                       for col_id in range(n_col) for p in ['timer_exc', 'messenger_exc', 'timer_inh', 'messenger_inh']],
        'record_conn_names':
            [(c, c) for col_id in range(n_col) for c in [naming.get_pop_name('timer_exc', col_id)]] +
            [('E_L5_{}'.format(i + 1), 'E_L23_{}'.format(i)) for i in range(n_col - 1)]
    }

    plotting_params = {
        'plot_single_traces': False,  # bool or integer
    }

    return dict([('kernel_pars', kernel_pars),
                 ('net_pars', net_params),
                 ('connection_pars', connection_params),
                 ('task_pars', task_params),
                 ('input_pars', input_params),
                 ('learning_pars', learning_params),
                 ('extractor_pars', extractor_parameters),
                 ('decoding_pars', decoding_parameters),
                 ('decoder_output_pars', decoder_outputs),
                 ('recording_pars', recording_params),
                 ('plotting_pars', plotting_params),
    ])

