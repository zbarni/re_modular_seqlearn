import numpy as np
import os
import sys
import itertools
import pickle
from typing import MutableMapping
import pandas as pd
from tqdm import tqdm
import string

try:
    import ntpath
except ImportError as e:
    raise ImportError("Import dependency not met: %s" % e)
import inspect
import errno

# FNA imports
from fna.tools.parameters import ParameterSet
from fna.tools.utils.data_handling import isiterable, import_mod_file
from fna.tools.utils import operations, logger

# add model source path
sys.path.append('/home/zbarni/code/projects/modseq/dev/')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from cone_shouval_src.defaults.paths import set_project_paths
from cone_shouval_src.utils.system import set_system_parameters

logs = logger.get_logger(__name__)
np.set_printoptions(threshold=sys.maxsize, suppress=True)


def parse_template(template_file, field_dictionary, save_to=None):
    """
    Read a template file and replace the provided fields
    :param template_file:
    :param field_dictionary:
    :param save_to:
    :return:
    """
    res = []

    with open(template_file, 'rb') as fp:
        tf = fp.readlines()
    for line in tf:
        new_line = line
        for k, v in list(field_dictionary.items()):
            new_line = new_line.replace(bytes(k, encoding='utf-8'), bytes(v, encoding='utf-8'))
        res.append(new_line)

    if save_to:
        with open(save_to, 'wb') as fp:
            fp.writelines(res)

    return res


def replace_experiment_labels(file, system_label=None, project_label=None):
    with open(file, 'rb') as fp:
        data = fp.readlines()
        for idx, line in enumerate(data):
            if system_label is not None and b"system_label" in line and line.index(b"system_label") == 0:
                data[idx] = "system_label = '{}'\n".format(system_label).encode()
            if project_label is not None and b"project_label" in line and line.index(b"project_label") == 0:
                data[idx] = "project_label = '{}'\n".format(project_label).encode()
    with open(file, 'wb') as fp:
        fp.writelines(data)


# ########################################################################################
class ParameterSpace:
    """
    A collection of `ParameterSets`, representing multiple points (combinations) in parameter space.
    Parses parameter scripts and runs experiments locally or creates job files to be run on a cluster.
    Can also harvest stored results from previous experiments and recreate the parameter space for post-profiler.
    """

    def __init__(self, initializer, system_label=None, project_label=None, keep_all=False):
        """
        Generate ParameterSpace containing a list of all ParameterSets
        :param initializer: file url
        :param keep_all: store all original parameters (??)
        :return: (tuple) param_sets, param_axes, global_label, size of parameter space
        """
        assert isinstance(initializer, str), "Filename must be provided"
        if system_label is not None or project_label is not None:
            replace_experiment_labels(initializer, system_label, project_label)
        with open(initializer, 'rb') as fp:
            self.parameter_file = fp.readlines()

        def validate_parameters_file(module):
            """
            Checks for any errors / incompatibilities in the structure of the parameter file. Function
            `build_parameters` is required, with or without arguments.
            :param module: imported parameter module object
            :return:
            """
            # TODO anything else?
            # build_parameters function must be defined!
            if not hasattr(module, "build_parameters"):
                raise ValueError("`build_parameters` function is missing!")

            # check parameter range and function arguments
            range_arguments = inspect.getfullargspec(module.build_parameters)[0]
            # logs.info(range_arguments)
            if not hasattr(module, "ParameterRange"):
                if len(range_arguments) == 0:
                    return
                raise ValueError("`ParameterRange` and arguments of `build_parameters` do not match!")

            for arg in range_arguments:
                if arg not in module.ParameterRange:
                    raise ValueError('ParameterRange variable `%s` is not in `ParameterRange` dictionary!' % arg)
                if not isiterable(module.ParameterRange[arg]):
                    raise ValueError('ParameterRange variable `%s` is not iterable! Should be list!' % arg)

        def validate_parameter_sets(param_sets):
            """
            :param param_sets:
            :return:
            """
            required_dicts = ["kernel_pars"]#, "encoding_pars", "net_pars"]
            for p_set in param_sets:
                for d in required_dicts:
                    if d not in p_set:
                        raise ValueError("Required parameter (dictionary) `%s` not found!" % d)

                # `data_prefix` required
                if "data_prefix" not in p_set["kernel_pars"]:
                    raise ValueError("`data_prefix` missing from `kernel_pars`!")

        def parse_parameters_file(url):
            """
            :param url:
            :return:
            """
            module_name, module_obj = import_mod_file(url)
            # print(sys.path)
            sys.path.pop(-1)
            del sys.modules[module_name]

            try:
                validate_parameters_file(module_obj)
            except ValueError as error:
                logs.info("Invalid parameter file! Error: %s" % error)
                exit(-1)

            range_args = inspect.getfullargspec(module_obj.build_parameters)[0]  # arg names in build_parameters function
            n_ranges = len(range_args)
            n_runs = int(np.prod([len(module_obj.ParameterRange[arg]) for arg in range_args]))  # nr combinations
            param_axes = dict()
            param_sets = []

            # build cross product of parameter ranges, sorting corresponds to argument ordering in build_parameters
            # empty if no range parameters are present: [()]
            range_combinations = list(itertools.product(*[module_obj.ParameterRange[arg] for arg in range_args]))
            # call build_parameters for each range combination, and pack returned values into a list (set)
            # contains a single dictionary if no range parameters
            param_ranges = [module_obj.build_parameters(*elem) for elem in range_combinations]
            global_label = param_ranges[0]['kernel_pars']['data_prefix']

            if n_ranges <= 3:
                # verify parameter integrity / completeness
                try:
                    validate_parameter_sets(param_ranges)
                except ValueError as error:
                    logs.info("Invalid parameter file! Error: %s" % error)

                # build parameter axes
                axe_prefixes = ['x', 'y', 'z']
                for range_index in range(n_runs):
                    params_label = global_label
                    for axe_index, axe in enumerate(axe_prefixes[:n_ranges]):  # limit axes to n_ranges
                        param_axes[axe + 'label'] = range_args[axe_index]
                        param_axes[axe + 'ticks'] = module_obj.ParameterRange[param_axes[axe + 'label']]
                        param_axes[axe + 'ticklabels'] = [str(xx) for xx in param_axes[axe + 'ticks']]
                        params_label += '_' + range_args[axe_index]
                        params_label += '=' + str(range_combinations[range_index][axe_index])

                    p_set = ParameterSet(param_ranges[range_index], label=params_label)
                    if not keep_all:
                        p_set = p_set.clean(termination='pars')
                    p_set.update({'label': params_label})
                    param_sets.append(p_set)
            else:
                # verify parameter integrity / completeness
                try:
                    validate_parameter_sets(param_ranges)
                except ValueError as error:
                    logs.info("Invalid parameter file! Error: %s" % error)

                # build parameter axes
                axe_prefixes = [a for a in string.ascii_letters][:n_ranges]
                for range_index in range(n_runs):
                    params_label = global_label
                    for axe_index, axe in enumerate(axe_prefixes[:n_ranges]):  # limit axes to n_ranges
                        param_axes[axe + 'label'] = range_args[axe_index]
                        param_axes[axe + 'ticks'] = module_obj.ParameterRange[param_axes[axe + 'label']]
                        param_axes[axe + 'ticklabels'] = [str(xx) for xx in param_axes[axe + 'ticks']]
                        params_label += '_' + range_args[axe_index]
                        params_label += '=' + str(range_combinations[range_index][axe_index])

                    p_set = ParameterSet(param_ranges[range_index], label=params_label)
                    if not keep_all:
                        p_set = p_set.clean(termination='pars')
                    p_set.update({'label': params_label})
                    param_sets.append(p_set)

            return param_sets, param_axes, global_label, n_ranges, module_obj.ParameterRange

        def parse_parameters_dict(url):
            """
            Simple parser for dictionary (text) parameter scripts, not .py!
            :param url:
            :return:
            """
            param_set = ParameterSet(url)
            try:
                validate_parameter_sets([param_set])
            except ValueError as error:
                logs.info("Invalid parameter file! Error: %s" % error)

            param_axes = {}
            label = param_set.kernel_pars["data_prefix"]
            dim = 1
            return [param_set], param_axes, label, dim, {}

        if initializer.endswith(".py"):
            self.parameter_sets, self.parameter_axes, self.label, self.dimensions, self.parameter_ranges = \
                parse_parameters_file(initializer)
        else:
            self.parameter_sets, self.parameter_axes, self.label, self.dimensions, self.parameter_ranges = \
                parse_parameters_dict(initializer)

    def update_run_parameters(self, system_label=None, project_label=None): #, system_params=None):
        """
        Update run type and experiments specific paths in case a system_label template is specified.
        :param project_label:
        :param system_label: name of system_label template, e.g., Blaustein
        :return:
        """
        paths = set_project_paths(system=system_label, project_label=project_label)
        # if system_params is None:
        #     system_params = set_system_parameters(cluster=system_label)
        # kernel_pars = set_kernel_defaults(resolution=0.1, run_type=system_label, data_label='',
        #                                   data_paths=paths, **system_params)

        if system_label is not None and system_label != "local":
            assert system_label in paths.keys(), "Default setting for cluster {0} not found!".format(system_label)
            for param_set in self.parameter_sets:
                param_set.kernel_pars['data_path'] = paths[system_label]['data_path']
                param_set.kernel_pars['mpl_path'] = paths[system_label]['matplotlib_rc']
                param_set.kernel_pars['print_time'] = False
                param_set.kernel_pars.system.local = False
                param_set.kernel_pars.system.system_label = system_label
                param_set.kernel_pars.system.jdf_template = paths[system_label]['jdf_template']
                param_set.kernel_pars.system.remote_directory = paths[system_label]['remote_directory']
                param_set.kernel_pars.system.queueing_system = paths[system_label]['queueing_system']
                # param_set.kernel_pars.system.jdf_fields = kernel_pars.system.jdf_fields
        else:
            for param_set in self.parameter_sets:
                param_set.kernel_pars['data_path'] = paths[system_label]['data_path']
                param_set.kernel_pars['mpl_path'] = paths[system_label]['matplotlib_rc']
                param_set.kernel_pars['print_time'] = True
                param_set.kernel_pars.system.local = True
                param_set.kernel_pars.system.system_label = system_label
                param_set.kernel_pars.system.jdf_template = paths[system_label]['jdf_template']
                param_set.kernel_pars.system.remote_directory = paths[system_label]['remote_directory']
                param_set.kernel_pars.system.queueing_system = paths[system_label]['queueing_system']

    def iter_sets(self):
        """
        An iterator which yields the ParameterSets in ParameterSpace
        """
        for val in self.parameter_sets:
            yield val

    def __eq__(self, other):
        """
        For testing purposes
        :param other:
        :return:
        """
        if not isinstance(other, self.__class__):
            return False

        for key in self.parameter_axes:
            if key not in other.parameter_axes:
                return False
            else:
                if isinstance(self.parameter_axes[key], np.ndarray):
                    if not np.array_equal(self.parameter_axes[key], other.parameter_axes[key]):
                        return False
                elif self.parameter_axes[key] != other.parameter_axes[key]:
                    return False

        if self.label != other.label or self.dimensions != other.dimensions:
            return False

        return self.parameter_sets == other.parameter_sets

    def __getitem__(self, item):
        return self.parameter_sets[item]

    def __len__(self):
        return len(self.parameter_sets)

    def save(self, target_full_path):
        """
        Save the full ParameterSpace by re-writing the parameter file
        :return:
        """
        with open(target_full_path, 'wb') as fp:
            fp.writelines(self.parameter_file)

    def compare_sets(self, parameter):
        """
        Determine whether a given parameter is common to all parameter sets
        :param parameter: parameter to compare
        """
        common = dict(pair for pair in self.parameter_sets[0].items() if all((pair in d.items() for d in
                                                                              self.parameter_sets[1:])))
        result = False
        if parameter in common.keys():
            result = True
        else:
            for k, v in common.items():
                if isinstance(v, dict):
                    if parameter in v.keys():
                        result = True
        return result

    def run(self, computation_function, project_dir=None, **parameters):
        """
        Run a computation on all the parameters
        :param computation_function: function to execute
        :param parameters: kwarg arguments for the function
        """
        system = self.parameter_sets[0].kernel_pars.system

        if system['local']:
            logs.info("Running {0} serially on {1} Parameter Sets".format(
                str(computation_function.__module__.split('.')[1]), str(len(self))))

            results = None
            for par_set in self.parameter_sets:
                logs.info("\n- Parameters: {0}".format(str(par_set.label)))
                results = computation_function(par_set, **parameters)
            return results
        else:
            logs.info("Preparing job description files...")
            export_folder = system['remote_directory']
            main_experiment_folder = export_folder + '{0}/'.format(self.label)

            try:
                os.makedirs(main_experiment_folder)
            except OSError as err:
                if err.errno == errno.EEXIST and os.path.isdir(main_experiment_folder):
                    logs.info("Path `{0}` already exists, will be overwritten!".format(main_experiment_folder))
                else:
                    raise OSError(err.errno, "Could not create exported experiments folder.", main_experiment_folder)

            remote_run_folder = export_folder + self.label + '/'
            project_dir = os.path.abspath(project_dir)
            network_dir = os.path.abspath(project_dir + '/../')
            py_file_common_header = ("import sys\nsys.path.append('{0}')\nsys.path.append('{1}')\nimport matplotlib"
                                     "\nmatplotlib.use('Agg')\nfrom fna.tools.parameters import *\nfrom "
                                     "fna.tools.analysis import *\nfrom experiments import {2}\n\n") \
                .format(project_dir, network_dir, computation_function.__module__.split('.')[1])

            write_to_submit = []
            job_list = main_experiment_folder + 'job_list.txt'

            for par_set in self.parameter_sets:
                system2 = par_set.kernel_pars.system
                template_file = system2['jdf_template']
                queueing_system = system2['queueing_system']
                exec_file_name = remote_run_folder + par_set.label + '.sh'
                local_exec_file = main_experiment_folder + par_set.label + '.sh'
                computation_file = main_experiment_folder + par_set.label + '.py'
                remote_computation_file = remote_run_folder + par_set.label + '.py'

                par_set.save(main_experiment_folder + par_set.label + '.txt')
                with open(computation_file, 'w') as fp:
                    fp.write(py_file_common_header)
                    fp.write(computation_function.__module__.split('.')[1] + '.' + computation_function.__name__ +
                             "({0}, **{1})".format("'./" + par_set.label + ".txt'", str(parameters)))

                system2['jdf_fields'].update({'{{ computation_script }}': remote_computation_file,
                                              '{{ script_folder }}': remote_run_folder})
                # print(system2)
                parse_template(template_file, system2['jdf_fields'].as_dict(), save_to=local_exec_file)
                write_to_submit.append("{0}\n".format(exec_file_name))

            with open(job_list, 'w') as fp:
                fp.writelines(write_to_submit)

    def print_stored_keys(self, data_path):
        """
        Print all the nested keys in the results dictionaries for the current data set
        :param data_path: location of "Results" folder
        :return:
        """
        def pretty(d, indent=1):
            if isinstance(d, dict):
                indent += 1
                for key, value in d.items():
                    print('-' * indent + ' ' + str(key))
                    if isinstance(value, dict):
                        pretty(value, indent + 1)

        pars_labels = [n.label for n in self]
        # open example data
        ctr = 0
        found_ = False
        while ctr < len(pars_labels) and not found_:
            try:
                with open(data_path + pars_labels[ctr], 'rb') as fp:
                    results = pickle.load(fp)
                print("Loading ParameterSet {0}".format(self.label))
                found_ = True
            except:
                print("Dataset {0} Not Found, skipping".format(pars_labels[ctr]))
                ctr += 1
                continue
        print("\n\nResults dictionary structure:\n- Root")
        pretty(results)

    def get_stored_keys(self, data_path):
        """
        Get a nested list of dictionary keys in the results dictionaries
        :return:
        """
        def extract_keys(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    for x in extract_keys(v):
                        yield "{}/{}".format(k, x)
                else:
                    yield k

        pars_labels = [n.label for n in self]
        # open example data
        ctr = 0
        found_ = False
        while ctr < len(pars_labels) and not found_:
            try:
                with open(data_path + pars_labels[ctr], 'rb') as fp:
                    results = pickle.load(fp)
                print("Loading ParameterSet {0}".format(self.label))
                found_ = True
            except:
                print("Dataset {0} Not Found, skipping".format(pars_labels[ctr]))
                ctr += 1
                continue
        return [k for k in extract_keys(results)]

    def harvest(self, data_path, key_set=None, operation=None, verbose=True):
        """
        Gather stored results data and populate the space spanned by the Parameters with the corresponding results
        :param data_path: full path to global data folder
        :param key_set: specific result to extract from each results dictionary (nested keys should be specified as
        'key1/key2/...'
        :param operation: function to apply to result, if any
        :return: (parameter_labels, results_array)
        """
        if self.parameter_axes:# and self.dimensions <= 3:
            # pars_path = data_path[:-8]+"ParameterSpace.py"
            # module_name, module_obj = import_mod_file(pars_path)
            # sys.path.pop(-1)
            # del sys.modules[module_name]
            # range_args = inspect.getfullargspec(module_obj.build_parameters)[0]
            # range_combinations = list(itertools.product(*[module_obj.ParameterRange[arg] for arg in range_args]))

            if self.dimensions <= 3:
                l = ['x', 'y', 'z']
            else:
                l = [a for a in string.ascii_letters][:self.dimensions]
            domain_lens = [len(self.parameter_axes[l[idx] + 'ticks']) for idx in range(self.dimensions)]
            domain_values = [self.parameter_axes[l[idx] + 'ticks'] for idx in range(self.dimensions)]
            var_names = [self.parameter_axes[l[idx] + 'label'] for idx in range(self.dimensions)]
            dom_len = np.prod(domain_lens)

            results_array = np.empty(tuple(domain_lens), dtype=object)
            parameters_array = np.empty(tuple(domain_lens), dtype=object)

            assert len(self) == dom_len, "Domain length inconsistent"
            pars_labels = [n.label for n in self]

            for n in tqdm(range(int(dom_len)), desc="Loading datasets", total=int(dom_len)):
                params_label = self.label
                index = []
                # for axe_index, axe in enumerate(l[:self.dimensions]):
                #     params_label += '_' + range_args[axe_index]
                #     params_label += '=' + str(range_combinations[range_index][axe_index])

                if self.dimensions >= 1:
                    idx_x = n % domain_lens[0]
                    params_label += '_' + var_names[0] + '=' + str(domain_values[0][idx_x])
                    index.append(idx_x)
                if self.dimensions >= 2:
                    idx_y = (n // domain_lens[0]) % domain_lens[1]
                    params_label += '_' + var_names[1] + '=' + str(domain_values[1][idx_y])
                    index.append(idx_y)
                if self.dimensions >= 3:
                    idx_z = ((n // domain_lens[0]) // domain_lens[1]) % domain_lens[2]
                    params_label += '_' + var_names[2] + '=' + str(domain_values[2][idx_z])
                    index.append(idx_z)
                if self.dimensions >= 4:
                    idx_a = ((n // domain_lens[0]) // domain_lens[1] // domain_lens[2]) % domain_lens[3]
                    params_label += '_' + var_names[3] + '=' + str(domain_values[3][idx_a])
                    index.append(idx_a)
                if self.dimensions >= 5:
                    idx_b = ((n // domain_lens[0]) // domain_lens[1] // domain_lens[2] // domain_lens[3]) % domain_lens[4]
                    params_label += '_' + var_names[4] + '=' + str(domain_values[4][idx_b])
                    index.append(idx_b)
                if self.dimensions >= 6:
                    idx_c = ((n // domain_lens[0]) // domain_lens[1] // domain_lens[2] // domain_lens[3] // domain_lens[4]) % domain_lens[5]
                    params_label += '_' + var_names[5] + '=' + str(domain_values[5][idx_c])
                    index.append(idx_c)
                if self.dimensions > 6:
                    raise NotImplementedError("Harvesting ParameterSpaces of more than 6 dimensions is not implemented yet")

                parameters_array[tuple(index)] = pars_labels[pars_labels.index(params_label)]
                try:
                    with open(data_path + params_label, 'rb') as fp:
                        results = pickle.load(fp)
                    if verbose:
                        print("Loading {0}".format(params_label))
                except:
                    if verbose:
                        print("Dataset {0} Not Found, skipping".format(params_label))
                    continue
                if key_set is not None:
                    try:
                        nested_result = NestedDict(results)
                        if operation is not None:
                            results_array[tuple(index)] = operation(nested_result[key_set])
                        else:
                            results_array[tuple(index)] = nested_result[key_set]
                    except:
                        continue
                else:
                    results_array[tuple(index)] = results
        else:
            parameters_array = self.label
            results_array = []
            try:
                with open(data_path + self.label, 'rb') as fp:
                    results = pickle.load(fp)
                if verbose:
                    print("Loading Dataset {0}".format(self.label))
                if key_set is not None:
                    nested_result = NestedDict(results)
                    assert isinstance(results, dict), "Results must be dictionary"
                    if operation is not None:
                        results_array = operation(nested_result[key_set])
                    else:
                        results_array = nested_result[key_set]
                else:
                    results_array = results
            except IOError:
                if verbose:
                    print("Dataset {0} Not Found, skipping".format(self.label))

        return parameters_array, results_array


class NestedDict(MutableMapping):
    """

    """

    def __init__(self, initial_value=None, root=True):
        super(self.__class__, self).__init__()
        self._val = {}
        if initial_value is not None:
            self._val.update(initial_value)
            self._found = False
        self._root = root

    def __getitem__(self, item):
        self._found = False

        def _look_deeper():
            result = tuple()
            for k, v in list(self._val.items()):
                if isinstance(v, dict):
                    n = NestedDict(self[k], root=False)
                    if n[item]:
                        result += (n[item],)
                    self._found = self._found or n._found
            if self._root:
                if self._found:
                    self._found = False
                else:
                    raise KeyError(item)
            result = result[0] if len(result) == 1 else result
            return result

        def _process_list():
            if len(item) == 1:
                return self[item[0]]
            trunk, branches = item[0], item[1:]
            nd = NestedDict(self[trunk], root=False)
            return nd[branches] if len(branches) > 1 else nd[branches[0]]

        if isinstance(item, list):
            return _process_list()
        elif self.__isstring_containing_char(item, '/'):
            item = item.split('/')
            return _process_list()
        elif item in self._val:
            self._found = True
            return self._val.__getitem__(item)
        else:
            return _look_deeper()

    def __setitem__(self, branch_key, value):
        self._found = False

        def _process_list():
            branches, tip = branch_key[1:], branch_key[0]
            if self[tip]:
                if isinstance(self[tip], tuple):
                    if isinstance(self[branches], tuple):
                        raise KeyError('ambiguous keys={!r}'.format(branch_key))
                    else:
                        self[branches][tip] = value
                else:
                    self[tip] = value
            else:
                raise KeyError('no key found={!r}'.format(tip))

        def _look_deeper():
            nd = NestedDict(root=False)
            for k, v in list(self._val.items()):
                if v and (isinstance(v, dict) or isinstance(v, NestedDict)):
                    nd._val = self._val[k]
                    nd[branch_key] = value
                    self._found = self._found or nd._found
            if self._root:
                if self._found:
                    self._found = False
                else:
                    self._val.__setitem__(branch_key, value)

        if isinstance(branch_key, list) or isinstance(branch_key, tuple):
            _process_list()
        elif self.__isstring_containing_char(branch_key, '/'):
            branch_key = branch_key.split('/')
            _process_list()
        elif branch_key in self._val:
            self._found = True
            self._val.__setitem__(branch_key, value)
        else:
            _look_deeper()

    def __delitem__(self, key):
        self._val.__delitem__(key)

    def __iter__(self):
        return self._val.__iter__()

    def __len__(self):
        return self._val.__len__()

    def __repr__(self):
        return __name__ + str(self._val)

    def __call__(self):
        return self._val

    def __contains__(self, item):
        return self._val.__contains__(item)

    def anchor(self, trunk, branch, value=None):
        nd = NestedDict(root=False)
        for k, v in list(self._val.items()):
            if v and isinstance(v, dict):
                nd._val = self._val[k]
                nd.anchor(trunk, branch, value)
                self._found = self._found or nd._found
            if k == trunk:
                self._found = True
                if not isinstance(self._val[trunk], dict):
                    if self._val[trunk]:
                        raise ValueError('value of this key is not a logical False')
                    else:
                        self._val[trunk] = {}  # replace None, [], 0 and False to {}
                self._val[trunk][branch] = value
        if self._root:
            if self._found:
                self._found = False
            else:
                raise KeyError

    def setdefault(self, key, default=None):
        if isinstance(key, list) or isinstance(key, tuple):
            trunk, branches = key[0], key[1:]
            self._val.setdefault(trunk, {})
            if self._val[trunk]:
                pass
            else:
                self._val[trunk] = default

            nd = NestedDict(self[trunk], root=False)
            if len(branches) > 1:
                nd[branches] = default
            elif len(branches) == 1:
                nd._val[branches[0]] = default
            else:
                raise KeyError
        else:
            self._val.setdefault(key, default)

    @staticmethod
    def __isstring_containing_char(obj, char):
        if isinstance(obj, str):
            if char in obj:
                return True
        return False
