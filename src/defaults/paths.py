import os
from pathlib import Path

"""
Defines paths to various directories and files specific to the running mode (locally or on a cluster).
Should be modified by users according to their particular needs.
"""


def set_project_paths(system, project_label):
	wd = os.path.abspath(Path(__file__).parent.parent)
	paths = {
		'local': {
			'data_path': 		wd + '/' + project_label + '/data/',  # output directory, must be created before running
			'jdf_template': 	None,								  # cluster template not needed
			'matplotlib_rc': 	wd + '/defaults/matplotlibrc',	      # custom matplotlib configuration
			'remote_directory': wd + '/' + project_label + '/data/export/',	# directory for export scripts to be run on cluster
			'queueing_system':  None},									# only when running on clusters
	}
	return {system: paths[system]}

