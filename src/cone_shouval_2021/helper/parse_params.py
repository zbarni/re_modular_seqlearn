"""
Parse synapse and neuron parameters.
"""
import copy
import logging
import nest


def get_synaptic_properties(parameters, synapse_model_name, reward_times=[], recurrent=True):
    """
    Compile synaptic properties into NEST parameter dictionaries.
    :param data:
    :param synapse_model_name:
    :param copy_synapse_model_name:
    :param reward_times:
    :param recurrent: [bool]
    :return: [list] 2 lists of NEST dictionaries containing the synaptic properties, plastic and static synapses
    """
    #plastic
    static = {'model': 'static_synapse', 'delay': 1., 'weight': 0.1, 'receptor_type': 0.}

    if synapse_model_name:
        copy_synapse_model_name = synapse_model_name + '_copy'
        learning_pars = parameters.learning_pars

        copymodel_syn_dict = {'the_delay': 1., 'T_reward': learning_pars.T_reward, 'T_tr': learning_pars.T_ref,
                              'reward_times': reward_times}

        copymodel_syn_dict['tau_w'] = learning_pars.tau_w

        nest.CopyModel(synapse_model_name, copy_synapse_model_name, copymodel_syn_dict)

        syn_spec_rec = {'w': .1234}
        syn_spec_ff = {'w': .1234}

        syn_spec_rec.update({'model': copy_synapse_model_name, 'Tp_max': learning_pars.Tp_max_rec,
                             'Td_max': learning_pars.Td_max_rec,
                             'tau_ltp': learning_pars.tau_ltp_rec, 'tau_ltd': learning_pars.tau_ltd_rec,
                             'eta_ltp': learning_pars.eta_ltp_rec, 'eta_ltd': learning_pars.eta_ltd_rec,
                             'learn_rate': learning_pars.learn_rate_rec, 'theta': learning_pars.rate_th_rec,
                             'weight_update_once': learning_pars.weight_update_once_rec,
                             'norm_traces':learning_pars.norm_traces_rec,
                             'weights_ns':learning_pars.weights_ns
                             })
        syn_spec_ff.update({'model': copy_synapse_model_name, 'Tp_max': learning_pars.Tp_max_ff,
                            'Td_max': learning_pars.Td_max_ff,
                            'tau_ltp': learning_pars.tau_ltp_ff, 'tau_ltd': learning_pars.tau_ltd_ff,
                            'eta_ltp': learning_pars.eta_ltp_ff, 'eta_ltd': learning_pars.eta_ltd_ff,
                            'learn_rate': learning_pars.learn_rate_ff, 'theta': learning_pars.rate_th_ff,
                            'weight_update_once': learning_pars.weight_update_once_ff,
                            'norm_traces':learning_pars.norm_traces_ff,
                            'weights_ns':learning_pars.weights_ns
                            })
        return [syn_spec_rec], [syn_spec_ff], [static]

    else:
        logging.warning("NO PLASTIC SYNAPSES !!! BE CAREFUL!")
        return [copy.deepcopy(static)], [copy.deepcopy(static)], [static]

