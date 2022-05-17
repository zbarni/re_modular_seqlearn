"""
Helper script to load and install NESTML modules.
"""

import nest


def get_module_to_install(model_name):
    """
    Maps the neuron / synapse model to the correct package module name
    :param model_name:
    :return:
    """
    name_map = {
        # neuron
        'ShouvalNeuronEtraceX': 'shouval_neuron_etracex_module',  # etrace
        'ShouvalNeuron': 'shouval_neuron_module', # esyn
        'ShouvalNeuronEuler': 'shouval_neuron_euler_module', #

        # synapse
        'shouval_etracex_synapse': 'shouval_etracex_synapse_module',  # etrace
        'shouval_multith_synapse': 'shouval_multith_synapse_module'  # esyn
    }
    return name_map[model_name]

def install_dynamic_modules(model_name):
    if model_name not in nest.Models():
        nest.Install(get_module_to_install(model_name))
