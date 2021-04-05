# ==================================================================
# import modules
# ==================================================================
import utils
import os, sys
import numpy as np
import tensorflow as tf
from args import args
import model as model

# ==================================================================
# import experiment settings
# ==================================================================
from experiments.unet import model_config as exp_config

# ==================================================================
# import general modules written by cscs for the hpc-predict project
# ==================================================================
current_dir_path = os.getcwd()
mr_io_dir_path = current_dir_path[:-39] + 'hpc-predict-io/python/'
sys.path.append(mr_io_dir_path)
from mr_io import FlowMRI, SegmentedFlowMRI

# ==================================================================
# import and setup logging
# ==================================================================
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ============================================================================
# input: image with shape  ---> [nx, ny, nz, nt, 4]
# output: segmentation probability of the aorta with shape  ---> [nx, ny, nz, nt]
# ============================================================================
def get_segmentation_probability(test_image,
                                 images_pl,
                                 training_pl,
                                 seg_prob_op,
                                 sess):
        
    # create an empty array to store the predicted segmentation probabilities
    predicted_seg_prob_aorta = np.zeros((test_image.shape[:-1]))
    
    # predict the segmentation one zz slice at a time
    for zz in range(test_image.shape[2]):
        predicted_seg_prob = sess.run(seg_prob_op, feed_dict = {images_pl: np.expand_dims(test_image[:,:,zz,:,:], axis=0), training_pl: False})
        predicted_seg_prob = np.squeeze(predicted_seg_prob) # squeeze out the added batch dimension
        predicted_seg_prob_aorta[:,:,zz,:] = predicted_seg_prob[:, :, :, 1] # the prob. of the FG is in the second channel
        
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
    modelname = 'best_dice.ckpt'
    try_checkpoint_paths = [
        os.path.join(args.training_output, exp_config.experiment_name, 'models', modelname),
        args.training_output
        ]
    best_dice_checkpoint_path = ''
    for chkp in try_checkpoint_paths:
        if os.path.exists(f'{chkp}.index'):
            logging.info(f'Found checkpoint at path={chkp}')
            best_dice_checkpoint_path = chkp
            break
        else:
            logging.info(f'Could not find checkpoint at path={chkp}')
    if not best_dice_checkpoint_path:
        logging.warning('Did not find a saved model. First need to run training successfully...')
        raise RuntimeError('No checkpoint available to restore from!')

    # ============================   
    # Loading data (a FlowMRI object written by Flownet and in the filename given by inference_input)
    # ============================   
    logging.info('================ Loading input FlowMRI from: ' + args.inference_input)    
    flow_mri = FlowMRI.read_hdf5(args.inference_input)        
    flowMRI_image = np.concatenate([np.expand_dims(flow_mri.intensity, -1), flow_mri.velocity_mean], axis=-1)  
    logging.info('shape of the test image before cropping / padding: ' + str(flowMRI_image.shape))
    # normalize
    flowMRI_image = utils.normalize_image(flowMRI_image)
    # crop / pad to common size
    orig_volume_size = flowMRI_image.shape[0:4]
    common_volume_size = [112, 112, 20, 24]
    flowMRI_image = utils.crop_or_pad_4dvol(flowMRI_image, common_volume_size)
    logging.info('shape of the test image after cropping / padding: ' + str(flowMRI_image.shape))
        
    # ============================
    # build the TF graph
    # ============================
    with tf.Graph().as_default():

        # ============================
        # create placeholders
        # ============================
        logging.info('================ Creating placeholders... ================')
        image_tensor_shape = [None] + list(exp_config.image_size) + [exp_config.nchannels] # e.g.[8, 112, 112, 25, 4] : [bs, nx, ny, nt, 4]
        images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name = 'images')
        training_pl = tf.placeholder(tf.bool, shape=[], name = 'training_or_testing')

        # ============================
        # build the graph that computes predictions from the segmentation cnn
        # ============================
        seg_logits, seg_softmax, seg_prediction = model.predict(images = images_pl,
                                                                model_handle = exp_config.model_handle,
                                                                training = training_pl,
                                                                nlabels = exp_config.nlabels)

        # ================================================================
        # Add init ops
        # ================================================================
        init_op = tf.global_variables_initializer()
                
        # ================================================================
        # create a saver that can be used to load the weights of the trained CNN
        # ================================================================
        train_vars_list = []
        for v in tf.trainable_variables():
            train_vars_list.append(v)
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
        saver_best_dice.restore(sess, best_dice_checkpoint_path)       

        # ============================
        # predict the segmentation probability for the image
        # ============================
        flowMRI_seg_prob = get_segmentation_probability(test_image = flowMRI_image,
                                                        images_pl = images_pl,
                                                        training_pl = training_pl,
                                                        seg_prob_op = seg_softmax,
                                                        sess = sess)
        logging.info('shape of predicted segmentation: ' + str(flowMRI_seg_prob.shape))

        # ============================
        # crop / pad back to the original dimensions
        # ============================
        flowMRI_seg_prob = np.expand_dims(flowMRI_seg_prob, axis = -1)
        flowMRI_seg_prob = utils.crop_or_pad_4dvol(flowMRI_seg_prob, orig_volume_size)
        flowMRI_seg_prob = np.squeeze(flowMRI_seg_prob)
        logging.info('shape of predicted segmentation: ' + str(flowMRI_seg_prob.shape))

        # ============================
        # create an instance of the SegmentedFlowMRI class, with the image information from flow_mri as well as the predicted segmentation probabilities
        # ============================
        segmented_flow_mri = SegmentedFlowMRI(flow_mri.geometry,
                                              flow_mri.time,
                                              flow_mri.time_heart_cycle_period,
                                              flow_mri.intensity,
                                              flow_mri.velocity_mean,
                                              flow_mri.velocity_cov,
                                              flowMRI_seg_prob)

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
