# ==================================================================
# import python modules
# ==================================================================
import os, sys
import time
import shutil
import logging
import numpy as np
import tensorflow as tf
import sklearn.metrics as met
from args import args

# ==================================================================
# import other modules written within the segmenter project
# ==================================================================
import utils
import model as model
from config.system import config as sys_config
import data_freiburg_numpy_to_hdf5
import data_flownet

# ==================================================================
# import general modules written by cscs for the hpc-predict project
# ==================================================================
# question: how to import from a directory several layers above the current one?
# for now, adding the path of the hpc-predict-io/python/ directory to sys.path
# ==================================================================
current_dir_path = os.getcwd()
mr_io_dir_path = current_dir_path[:-39] + 'hpc-predict-io/python/'
sys.path.append(mr_io_dir_path)
from mr_io import FlowMRI, SegmentedFlowMRI
from scipy.interpolate import RegularGridInterpolator

# ==================================================================
# import experiment settings
# ==================================================================
from experiments.unet import model_config as exp_config

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# set logging directory according to the parsed arguments (training / doing inference)
# ==================================================================
if args.train is True and args.training_output is not None:
    log_dir = args.training_output
    
elif args.train is False and args.inference_output is not None:
    log_dir = args.inference_output
    
else:
    log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)

logging.info('================ Logging directory: %s ================' %log_dir)

# ============================================================================
# def get_segmentation
# input: image with shape  ---> [nz, nx, ny, nt, 4]
# output: segmentation probability with shape  ---> [nz, nx, ny, nt, 2]
# ============================================================================
def get_segmentation(image):
        
    # create an empty array to store the predicted segmentation probabilities
    predicted_segmentation = np.zeros((image.shape[:-1]))    
    # predict the segmentation one zz slice at a time
    for zz in range(image.shape[0]):
        predicted_segmentation[zz:zz+1,...] = sess.run(seg_mask, feed_dict = {images_pl: image[zz:zz+1,...]})
        
    return predicted_segmentation

# ==================================================================
# main function for inference
# ==================================================================
def run_inference():

    # ============================
    # log experiment details
    # ============================
    logging.info('============================================================')
    logging.info('================ EXPERIMENT NAME: %s ================' % exp_config.experiment_name)
    logging.info('================ Predicting segmentation for this image: %s ================' % args.inference_input)

    # ============================
    # load saved checkpoint file, if available
    # ============================
    logging.info('================ Looking for saved segmentation model... ================')
    if os.path.exists(os.path.join(args.training_output,'models/best_dice.ckpt')):
        best_dice_checkpoint_path = os.path.join(args.training_output, 'models/best_dice.ckpt')
        logging.info('Found saved model at %s. This will be used for predicted the segmentation.' % best_dice_checkpoint_path)
    else:
        logging.warning('Did not find a saved model. First need to run training successfully...')
        raise RuntimeError('No checkpoint available to restore from!')

    # ============================   
    # Loading data (from an hpc-predict-io MRI)
    # Q1. What kind of a hdf5 file does FlowMRI.read_hdf5 expect?
    # That is, what fields does this hdf5 file need to contain?
    # How to write such a hdf5 file from the Freiburg .dicom files? 
    # How to write such a hdf5 file from the Flownet .mat files? 
    # ============================   
    logging.info('================ Loading input FlowMRI from: %s ================' + args.inference_input)
    flow_mri = FlowMRI.read_hdf5(args.inference_input)

    # ============================
    # Extracting the image information required for the segmentation cnn
    # ============================
    image = np.concatenate([np.expand_dims(flow_mri.intensity, -1), flow_mri.velocity_mean], axis=-1).transpose([3,0,1,2,4])
    
    # ============================
    # Q2. What is the purpose of RegularGridInterpolator?
    # Going back to the construction of the FlowMRI object, will the values in flow_mri.geometry match the image size expected by the cnn (exp_config.image_size)?
    # If not, is this interpolation a way to resolve this size mismatch?
    # ============================
    image_preprocessed = np.zeros((image.shape[0], *exp_config.image_size, image.shape[-1])) 
    
    # ============================
    # TODO: exact coordinate transformation
    # ============================
    np.array(np.meshgrid([0,1], [2,3], indexing='ij')).transpose([1, 2, 0])
    target_points = np.array(np.meshgrid(np.linspace(flow_mri.geometry[0][0], flow_mri.geometry[0][-1], exp_config.image_size[0]),
                                         np.linspace(flow_mri.geometry[1][0], flow_mri.geometry[1][-1], exp_config.image_size[1]),
                                         np.linspace(flow_mri.geometry[2][0], flow_mri.geometry[2][-1], exp_config.image_size[2]),
                                         indexing='ij')).transpose([1,2,3,0])
    for t in range(image.shape[0]):
        for v in range(image.shape[-1]):
            rgi = RegularGridInterpolator(points = tuple(flow_mri.geometry), values = image[t, ..., v])
            image_preprocessed[t,...,v] = rgi(target_points)
    image = image_preprocessed
    del image_preprocessed

    logging.info('================ Shape of the image to be segmented: %s ================' %str(image.shape)) 
        
    # ============================
    # build the TF graph
    # ============================
    with tf.Graph().as_default():

        # ============================
        # create placeholders
        # ============================
        logging.info('================ Creating placeholders... ================')
        image_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size) + [exp_config.nchannels]
        images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name = 'images')

        # ============================
        # Build the graph that computes predictions from the segmentation cnn
        # ============================
        seg_logits, seg_softmax, seg_prediction = model.predict(images_pl,
                                                                exp_config.model_handle)

        # ============================
        # additional ops for converting the logits into probabilities.
        # Q3. why was this added additionally to the softmax probabilities obtained above?
        # ============================
        logits_tensor_shape = list(flow_mri.intensity.shape[:3]) + [exp_config.nlabels]
        logits_pl = tf.placeholder(tf.float32, shape = logits_tensor_shape)
        seg_probs = tf.nn.softmax(logits_pl)

        # ================================================================
        # Add init ops
        # ================================================================
        init_op = tf.global_variables_initializer()
        
        # ================================================================
        # Create savers for each domain
        # ================================================================
        max_to_keep = 15
        saver = tf.train.Saver(max_to_keep = max_to_keep)
        saver_best_dice = tf.train.Saver()

        # ================================================================
        # Create session
        # ================================================================
        sess = tf.Session()

        # ================================================================
        # freeze the graph before execution
        # ================================================================
        logging.info('================ Freezing the graph now! ================')
        tf.get_default_graph().finalize()

        # ================================================================
        # Run the Op to initialize the variables.
        # ================================================================
        logging.info('================ Initializing all variables ================')
        sess.run(init_op)
        
        # ================================================================
        # Restore session from a saved checkpoint
        # ================================================================
        logging.info('================ Restoring session from: %s ================' % best_dice_checkpoint_path)
        saver.restore(sess, best_dice_checkpoint_path)

        # ============================
        # predict the predicted logits
        # ============================
        # first, create an empty array in which the predicted segmentation logits will be populated
        image_seg_logits = np.zeros(shape=(*image.shape[:-1], exp_config.nlabels))    

        # go through each zz slice, one at a time and do the predicted for the entire 4D volume
        for b_i in range(0, image.shape[0], 1):
            batch_indices = np.arange(b_i, np.min([b_i + 1, n_images]))
            seg_logits_tmp = sess.run(seg_logits, feed_dict = {images_pl: image[batch_indices, ...]})
            image_seg_logits[batch_indices, ...] = seg_logits_tmp[0][...]

        # ================================================================
        # ================================================================
        target_points = np.array(np.meshgrid(*flow_mri.geometry, indexing='ij')).transpose([1, 2, 3, 0])
        image_seg_probs_mri = np.zeros((flow_mri.intensity.shape[3], *flow_mri.intensity.shape[:3]))
        image_seg_logits_mri = np.zeros((*flow_mri.intensity.shape[:3], exp_config.nlabels))

        # ============================
        # Q4. What is happening in this part? 
        # Going through each zz slice and channel of the predicted logits, are they being interpolated back to the shape of the original image?
        # ============================
        for t in range(image_seg_logits.shape[0]):
            for v in range(image_seg_logits.shape[-1]):
                rgi = RegularGridInterpolator(points = (np.linspace(flow_mri.geometry[0][0], flow_mri.geometry[0][-1], exp_config.image_size[0]),
                                               np.linspace(flow_mri.geometry[1][0], flow_mri.geometry[1][-1], exp_config.image_size[1]),
                                               np.linspace(flow_mri.geometry[2][0], flow_mri.geometry[2][-1], exp_config.image_size[2])),
                                               values = image_seg_logits[t, ..., v])
                
                image_seg_logits_mri[..., v] = rgi(target_points)
            
            image_seg_probs_mri[t, ...] = sess.run(seg_probs, feed_dict = {logits_pl: image_seg_logits_mri})[..., 0]
        image_seg_probs = image_seg_probs_mri
        del images_tr_seg_probs_mri

        # ============================
        # create an instance of the SegmentedFlowMRI class, with the image information from flow_mri as well as the predicted segmentation probabilities
        # ============================
        segmented_flow_mri = SegmentedFlowMRI(flow_mri.geometry,
                                              flow_mri.time,
                                              flow_mri.time_heart_cycle_period,
                                              flow_mri.intensity,
                                              flow_mri.velocity_mean,
                                              flow_mri.velocity_cov,
                                              image_seg_probs.transpose([1,2,3,0])[...])

        # ============================
        # write SegmentedFlowMRI to file
        # ============================
        inference_output_file = os.path.join(args.inference_output, os.path.basename(args.inference_input)[:-3] + '_cnn_segmented.h5')  
        logging.info('================ Writing SegmentedFlowMRI to: %s ================' + inference_output_file)
        segmented_flow_mri.write_hdf5(inference_output_file)
        logging.info('============================================================')