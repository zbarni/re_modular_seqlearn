"""
Main experiment script to create the model and run the simulations.
Figures are created during postprocessing via other scripts.
"""

import pickle
import os

from fna.tools.visualization.helper import set_global_rcParams
from fna.tools.utils import logger
from fna.tools.utils.data_handling import set_storage_locations

from helper import network_setup, log_time
from helper import input_seq as input_helper
from helper import simulation
from utils.parameters import ParameterSet

logprint = logger.get_logger(__name__)


def run(parameters, display=False, plot=True, save=True, load_inputs=False):
    # ############################ SYSTEM
    # experiments parameters
    if not isinstance(parameters, ParameterSet):
        parameters = ParameterSet(parameters)

    storage_paths = set_storage_locations(parameters.kernel_pars.data_path, parameters.kernel_pars.data_prefix,
                                          parameters.label, save=save)

    logger.update_log_handles(job_name=parameters.label, path=storage_paths['logs'])
    log_time(logger, timer_label="simulation", experiment_label=parameters.label, start=True, stop=False)
    log_time(logger, timer_label="prepare", experiment_label=parameters.label, start=True, stop=False)

    # create network
    snn = network_setup.build_network(parameters)

    # init input
    encoder, sequencer, embedded_signal, intervals = input_helper.fna_init_input(parameters, snn)

    # simulate train and test batches
    reward_times, recorded_data = simulation.simulate(parameters, snn, encoder, sequencer, embedded_signal, intervals,
                                                      storage_paths)

    if save:
        # save parameters
        parameters.save(storage_paths['parameters'] + parameters.label)

        # save inputs
        sequencer.save(parameters.label)
        embedded_signal.save(parameters.label)

        filename = 'Recordings_{}.data'.format(parameters.label)
        with open(os.path.join(storage_paths['activity'], filename), 'wb') as f:
            data = {
                'reward_times': reward_times,
                'recorded_data': recorded_data
            }
            pickle.dump(data, f)

    log_time(logger, timer_label="process", experiment_label=parameters.label, start=False, stop=True)
    log_time(logger, timer_label="simulation", experiment_label=parameters.label, start=False, stop=True)
    logger.log_stats(cmd=None, flush_file='{}.log'.format(parameters.label),
                     flush_path=os.path.join(parameters.kernel_pars.data_path + parameters.kernel_pars.data_prefix,
                                             'logs'))
