import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# import argparse
#
# # Parse data input and output directories
# def parse_args():
#     # Parse arguments
#     parser = argparse.ArgumentParser(description='Run CNN Segmenter for 4D MRI.')
#     parser.add_argument('--config', type=str, default='config/cnn_segmenter_cscs.json', # default='system/cnn_segmenter_neerav.json',
#                     help='Directory containing MRI data set')
#     return parser.parse_args()
#
# args = parse_args()

from args import args

class Config:
    pass

config = Config()

import json
with open(args.config, 'r') as f:
    conf_dict = json.load(f)
    for k,v in conf_dict.items():
        setattr(config, k, v)

# # ==================================================================
# # SET THESE PATHS MANUALLY
# # ==================================================================
#
# # ==================================================================
# # name of the host - used to check if running on cluster or not
# # ==================================================================
# local_hostnames = ['bmicdl05']
#
# # ==================================================================
# # project dirs
# # ==================================================================
# project_code_root = '/usr/bmicnas01/data-biwi-01/nkarani/students/nicolas/code/segmenter_cnn'
# project_data_root = '/usr/bmicnas01/data-biwi-01/nkarani/students/nicolas/data/freiburg'
# # Note that this is the base direectory where the freiburg images have been saved a numpy arrays.
# # The original dicom files are not here and they are not required for any further processing.
# # The base path for the original dicom files of the freiburg dataset are here:
# orig_data_root = '/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/data/freiburg/main_data_transfer/126_subjects'
#
# # ==================================================================
# # log root
# # ==================================================================
# log_root = os.path.join(project_code_root, 'logdir/v0.1/')
