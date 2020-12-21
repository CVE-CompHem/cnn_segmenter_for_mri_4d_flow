# ==================================================================
# import python modules
# ==================================================================
import os, sys
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from args import args

# ==================================================================
# import other modules written within the segmenter project
# ==================================================================
import model as model

# ==================================================================
# import experiment settings
# ==================================================================
from experiments.unet import model_config as exp_config

# ==================================================================
# import paths
# ==================================================================
# Some fixed paths can be read from here, if required.
# The paths read here are set in the file given in args.config
# Particularly, the path of the saved model can be read from here.
# Right now, it is read from the file path set in args.training_output
# ==================================================================
# from config.system import config as sys_config 

# ==================================================================
# needed these while importing data using the data loading scripts written by CVL
# Now, using data loading I/O interface written by CSCS
# ==================================================================
# import data_freiburg_numpy_to_hdf5
# import data_flownet

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
# import and setup logging
# ==================================================================
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ============================================================================
# input: image with shape  ---> [nz, nx, ny, nt, 4]
# output: segmentation probability of the aorta with shape  ---> [nz, nx, ny, nt]
# ============================================================================
def get_segmentation_probability(test_image,
                                 images_pl,
                                 seg_prob_op,
                                 sess):
        
    # create an empty array to store the predicted segmentation probabilities
    predicted_seg_prob_aorta = np.zeros((test_image.shape[:-1]))
    
    # predict the segmentation one zz slice at a time
    for zz in range(test_image.shape[0]):
        predicted_seg_prob = sess.run(seg_prob_op, feed_dict = {images_pl: test_image[zz:zz+1, ...]})
        predicted_seg_prob = np.squeeze(predicted_seg_prob)
        predicted_seg_prob_aorta[zz:zz+1,...] = predicted_seg_prob[:, :, :, 1]
        
    return predicted_seg_prob_aorta

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
    logging.info('================ Looking for saved segmentation model... ')
    logging.info('args.training_output: ' + args.training_output  + exp_config.experiment_name + '/models/best_dice.ckpt-20000.index')
    if os.path.exists(args.training_output + exp_config.experiment_name + '/models/best_dice.ckpt-20000.index'):
        best_dice_checkpoint_path = args.training_output + exp_config.experiment_name + '/models/best_dice.ckpt-20000.index'
        logging.info('Found saved model at %s. This will be used for predicted the segmentation.' % best_dice_checkpoint_path)
    else:
        logging.warning('Did not find a saved model. First need to run training successfully...')
        raise RuntimeError('No checkpoint available to restore from!')

    # ============================   
    # Loading data (a FlowMRI object written by Flownet and in the filename given by inference_input)
    # ============================   
    logging.info('================ Loading input FlowMRI from: ' + args.inference_input)
    flow_mri = FlowMRI.read_hdf5(args.inference_input)
    logging.info('shape of input intensity: ' + str(flow_mri.intensity.shape))
    logging.info('shape of velocity mean: ' + str(flow_mri.velocity_mean.shape))

    # ============================
    # Extracting the image information required for the segmentation cnn
    # ============================
    test_image = np.concatenate([np.expand_dims(flow_mri.intensity, -1), flow_mri.velocity_mean], axis=-1)    
    # ============================
    # Move the axes around so that we have [nz, nx, ny, nt, num_channels]
    # TODO: Check if the data orientation is similiar to what the CNN was trained on
    # ============================
    test_image = test_image.transpose([3, 0, 1, 2, 4])
    logging.info('shape of the test image before resampling: ' + str(test_image.shape))

    # ============================
    # Resample the image, so that we have the same shape as the one for which the CNN was trained (exp_config.image_size).
    # TODO: Instead of changing the resolution directly to match the sizes,
    # bring the test images to the resolution of the training images,
    # and then pad with zeros or crop.
    # ============================
    test_image_resampled = np.zeros((test_image.shape[0], *exp_config.image_size, test_image.shape[-1])) 
    target_points = np.array(np.meshgrid(np.linspace(flow_mri.geometry[0][0], flow_mri.geometry[0][-1], exp_config.image_size[0]),
                                         np.linspace(flow_mri.geometry[1][0], flow_mri.geometry[1][-1], exp_config.image_size[1]),
                                         np.linspace(flow_mri.geometry[2][0], flow_mri.geometry[2][-1], exp_config.image_size[2]),
                                         indexing='ij')).transpose([1,2,3,0])
    logging.info('initial target points: ' + str(target_points.shape))

    for t in range(test_image.shape[0]):
        for v in range(test_image.shape[-1]):
            rgi = RegularGridInterpolator(points = tuple(flow_mri.geometry), values = test_image[t, ..., v])
            test_image_resampled[t,...,v] = rgi(target_points)
    logging.info('shape of image after resampling: ' + str(test_image_resampled.shape))
        
    # ============================
    # build the TF graph
    # ============================
    with tf.Graph().as_default():

        # ============================
        # create placeholders
        # ============================
        logging.info('================ Creating placeholders... ================')
        image_tensor_shape = [None] + list(exp_config.image_size) + [exp_config.nchannels]
        images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name = 'images')

        # ============================
        # build the graph that computes predictions from the segmentation cnn
        # ============================
        seg_logits, seg_softmax, seg_prediction = model.predict(images = images_pl,
                                                                model_handle = exp_config.model_handle,
                                                                nlabels = exp_config.nlabels)
        logging.info('model built')

        # ================================================================
        # Add init ops
        # ================================================================
        init_op = tf.compat.v1.global_variables_initializer()
        
        # ================================================================
        # Make list of all trainable variables
        # ================================================================
        train_vars_list = []
        for v in tf.compat.v1.trainable_variables():
            train_vars_list.append(v)            
            print(v.name)
        
        # ================================================================
        # create a saver that can be used to load the weights of the trained CNN
        # ================================================================
        saver_best_dice = tf.compat.v1.train.Saver(var_list = train_vars_list)

        # ================================================================
        # Create session
        # ================================================================
        sess = tf.compat.v1.Session()

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
        # TODO: Ensure that the saved model is compatible with the model built here.
        # saver_best_dice.restore(sess, best_dice_checkpoint_path)       

        # ============================
        # predict the segmentation probability for the image
        # ============================
        test_image_seg_prob_resampled = get_segmentation_probability(test_image = test_image_resampled,
                                                                     images_pl = images_pl,
                                                                     seg_prob_op = seg_softmax,
                                                                     sess = sess)
        logging.info('shape of predicted segmentation before resampling: ' + str(test_image_seg_prob_resampled.shape))

        # ================================================================
        # resample the predicted segmentation back to the original shape
        # ================================================================
        test_image_seg_prob = np.zeros((flow_mri.intensity.shape[3], *flow_mri.intensity.shape[:3]))
        for zz in range(test_image_seg_prob_resampled.shape[0]):
            test_image_seg_prob[zz, ...] = resize(image = test_image_seg_prob_resampled[zz, ...],
                                                  output_shape = [test_image.shape[1], test_image.shape[2], test_image.shape[3]],
                                                  order = 1,
                                                  preserve_range = True,
                                                  anti_aliasing = True)
        logging.info('shape of predicted segmentation after resampling back to the original resolution: ' + str(test_image_seg_prob.shape))
        
        # ================================================================
        # Reorder the axes to match that of the original data
        # ================================================================
        test_image_seg_prob = test_image_seg_prob.transpose([1, 2, 3, 0])
        logging.info('shape of predicted segmentation after resampling back to the original resolution: ' + str(test_image_seg_prob.shape))

        # ============================
        # create an instance of the SegmentedFlowMRI class, with the image information from flow_mri as well as the predicted segmentation probabilities
        # ============================
        segmented_flow_mri = SegmentedFlowMRI(flow_mri.geometry,
                                              flow_mri.time,
                                              flow_mri.time_heart_cycle_period,
                                              flow_mri.intensity,
                                              flow_mri.velocity_mean,
                                              flow_mri.velocity_cov,
                                              test_image_seg_prob)

        # ============================
        # write SegmentedFlowMRI to file
        # ============================
        logging.info('================ Writing SegmentedFlowMRI to: ' + args.inference_output)
        segmented_flow_mri.write_hdf5(args.inference_output)
        logging.info('============================================================')

# ==================================================================
# ==================================================================
def main():
    
    run_inference()

# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()