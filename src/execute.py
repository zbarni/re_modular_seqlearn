#!/usr/bin/python
import os
import sys

n_threads = 8
os.environ['MKL_NUM_THREADS'] = str(n_threads)
os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
os.environ['NUMEXPR_MAX_THREADS'] = str(n_threads)
os.environ['OMP_NUM_THREADS'] = str(n_threads)

import copy

import matplotlib
from argparse import ArgumentParser
import importlib
from utils.parameters import ParameterSpace
from fna.tools.utils import logger

cmdl_parse = copy.copy(sys.argv)

logs = logger.get_logger(__name__)


def run_experiment(project_label, parameter_file, experiment="noise_driven_dynamics", system=None,
                   **parameters):
    """
    Entry point, parses parameters and runs experiments locally or creates scripts to be run on a cluster.
    :param project_label:
    :param parameter_file: full path to parameter file
    :param experiment: which experiments to run
    :param system: system label. Corresponding entry must be in defaults.paths!
    :param parameters: other CLI input parameters
    :return:
    """
    logger.update_log_handles(main=True)

    try:
        # determine the project folder and add it to sys.path
        # project_dir, _ = path.split(path.split(parameter_file)[0])
        project_dir = os.getcwd() + "/" + project_label
        sys.path.append(project_dir)
        experiment_mod = importlib.import_module("experiments." + experiment)
    except Exception as err:
        logs.warning("Could not find experiment `{0}`. Is it in the project's ./experiments/ directory? \nError: {1}".format(
            experiment, str(err)))
        exit(-1)

    logger.log_timer.start("parameters")

    pars = ParameterSpace(parameter_file, system_label=system, project_label=project_label)
    pars.update_run_parameters(system_label=system, project_label=project_label)

    if not os.path.exists(pars[0].kernel_pars.data_path+pars[0].kernel_pars.data_prefix+'/'):
        os.makedirs(pars[0].kernel_pars.data_path+pars[0].kernel_pars.data_prefix+'/')

    if 'overwrite_parameter_file' not in parameters or parameters['overwrite_parameter_file']:
        pars.save(pars[0].kernel_pars.data_path+pars[0].kernel_pars.data_prefix+'/'+'ParameterSpace.py')
    if 'overwrite_parameter_file' in parameters:
        del parameters['overwrite_parameter_file']  # remove as it's not needed later

    logger.log_timer.stop("parameters")
    if pars[0].kernel_pars.system['local']:
        logger.log_timer.start("experiments")
        pars.run(experiment_mod.run, **parameters)
        logger.log_timer.stop("experiments")
    else:
        logger.log_timer.start("cluster job preparation")
        pars.run(experiment_mod.run, project_dir, **parameters)
        logger.log_timer.stop("cluster job preparation")

    logger.log_stats(cmd=str.join(' ', cmdl_parse), flush_file='{}_submission.log'.format(pars[0].kernel_pars.data_prefix),
                     flush_path=os.path.join(pars[0].kernel_pars.data_path+pars[0].kernel_pars.data_prefix, 'logs'))


def print_welcome_message(project_label, experiment_label, run_label):
    print("""
***************************************
*** Functional Neural Architectures ***
***************************************
- Project: {0}
- Experiment: {1}
- Parameters: {2}
""".format(project_label, experiment_label, run_label))


def create_parser():
    """
    Create command line parser with options.
    :return: ArgumentParser
    """
    parser_ = ArgumentParser(prog="execute.py")
    parser_.add_argument("--project", dest="project", nargs=1, default=None, metavar="project_label",
                         help="project label", required=True)
    parser_.add_argument("-p", dest="p_file", nargs=1, default=None, metavar="parameter_file",
                         help="absolute path to parameter file", required=True)
    parser_.add_argument("-c", dest="c_function", nargs=1, default=None, metavar="experiment_function",
                         help="experiments to execute", required=True)
    parser_.add_argument("--system", dest="cluster", nargs=1, default=None, metavar="Blaustein",
                         help="name of cluster entry in default paths dictionary")
    parser_.add_argument("--extra", dest="extra", nargs='*', default=[], metavar="extra arguments",
                         help="extra keyword arguments for the experiments")
    return parser_


if __name__ == "__main__":
    args = create_parser().parse_args(cmdl_parse[1:])
    print_welcome_message(project_label=args.project[0], experiment_label=args.c_function[0], run_label=args.p_file[0])

    d = dict([arg.split('=', 1) for arg in args.extra])
    for k, v in d.items():
        if v == 'False':
            d.update({k: False})
        elif v == 'True':
            d.update({k: True})
        elif v == 'None':
            d.update({k: None})

    cluster = args.cluster[0] if args.cluster is not None else "local"
    run_experiment(project_label=args.project[0], parameter_file=args.p_file[0],
                   experiment=args.c_function[0], system=cluster, **d)