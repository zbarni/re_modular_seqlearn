from fna.tools.parameters import ParameterSet, extract_nestvalid_dict


def set_system_parameters(cluster, queue='batch', walltime='00-24:00:00', nodes=1, ppn=48, mem=64000):
    if cluster == 'local':
        system = dict(
            nodes=1,  # number of nodes
            ppn=16,  # number of cores / node (max 20)
            walltime='01-00:00:00',
            queue='batch')
    else:
        system = {}
    return system


def reset_nest_kernel(kernel_pars):
    import nest

    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.SetKernelStatus(extract_nestvalid_dict(kernel_pars.as_dict(), param_type='kernel'))

# def set_storage_paths(project_label, system_label, experiment_label):
#     """
#
#     :param project_label:
#     :param system_label:
#     :param experiment_label:
#     :return:
#     """
#     return {system_label: io.set_storage_locations(data_path, data_label, instance_label, save=save)}