# ========================================================================
# Read the images and segmentations
# Preprocess them: Normalize, and make them of the same size by cropping / padding with zeros
# Save them together as a training_data.hdf5 file
# ========================================================================

# ============================   
# import module and set paths
# ============================   
import numpy as np
import utils
import matplotlib.pyplot as plt
import h5py

# ============================   
# adding io interface developed by cscs
# Question: how to import from a directory several layers above the current one?
# For now, adding the path of the hpc-predict-io/python/ directory to sys.path
# ============================   
import os, sys
current_dir_path = os.getcwd()
mr_io_dir_path = current_dir_path[:-39] + 'hpc-predict-io/python/'
sys.path.append(mr_io_dir_path)
from mr_io import FlowMRI, SegmentedFlowMRI

# ============================   
# Basepath where the training data is stored as individual files
# ============================   
basepath = "/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/data/eth_ibt/flownet/pollux/all_data/"

# ============================   
# The size to which all volumes should be cropped / padded into
# ============================   
common_volume_size = [112, 112, 20, 24]

# training on the first 3 subjects, on the fully sampled images, and on undersampled values upto 18
subjects = [1, 2, 3]
num_subjects = len(subjects)
R_values = [1, 8, 10, 12, 14, 16, 18]
num_R_values = len(R_values)
num_images = num_subjects * num_R_values

# =======================
# Create a hdf5 file where all training images and ground truth segmentations will be written
# =======================
training_data_filename = basepath + "training_data.hdf5"
hdf5_file = h5py.File(training_data_filename, "w")

# =======================
# Create empty array for storing images and labels
# =======================
data = {}
data['images'] = hdf5_file.create_dataset("images", [num_images] + list(common_volume_size) + [4], dtype=np.float32)
data['labels'] = hdf5_file.create_dataset("labels", [num_images] + list(common_volume_size) + [1], dtype=np.float32)

for subnum in range(len(subjects)):

    for r in range(len(R_values)):

        sub = subjects[subnum]
        R = R_values[r]
        
        # read the image for this subject and R value
        flowmripath = basepath + 'v' + str(sub) + '_R' + str(R) + '.h5'
        flow_mri = FlowMRI.read_hdf5(flowmripath)        
        flowMRI_image = np.concatenate([np.expand_dims(flow_mri.intensity, -1), flow_mri.velocity_mean], axis=-1)  
        # normalize
        flowMRI_image = utils.normalize_image(flowMRI_image)
        # crop / pad to common size
        flowMRI_image = utils.crop_or_pad_4dvol(flowMRI_image, common_volume_size)
        
        # read the segmentation for this subject 
        segmentedflowmripath = basepath + 'v' + str(sub) + '_seg_rw.h5'
        segmented_flow_mri = SegmentedFlowMRI.read_hdf5(segmentedflowmripath)
        flowMRI_seg = np.expand_dims(segmented_flow_mri.segmentation_prob, -1)
        # crop / pad to common size
        flowMRI_seg = utils.crop_or_pad_4dvol(flowMRI_seg, common_volume_size)

        # For debugging: check if the image and segmentation look okay
        visualize = False
        if visualize:
            plt.figure()
            plt.imshow(flowMRI_image[:,:,8,3,0],'gray', alpha=0.99)
            plt.imshow(flowMRI_seg[:,:,8,3,0],'gray', alpha=0.5)
            plt.show()
            
        id_this_subject = subnum*num_R_values + r
        print(id_this_subject)
        data['images'][id_this_subject:id_this_subject+1, ...] = flowMRI_image
        data['labels'][id_this_subject:id_this_subject+1, ...] = flowMRI_seg

# Close the hdf5 file containing all the training data
hdf5_file.close()
