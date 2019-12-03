# ==================================================================
# import 
# ==================================================================
import time
import shutil
import logging
import os.path
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import utils
import model as model
from config.system import config as sys_config
import data_freiburg_numpy_to_hdf5
# import augment_data_unet as ad

# import warnings
# warnings.filterwarnings('ignore', '.*output shape of zoom.*')

from args import args

from mr_io import FlowMRI, SegmentedFlowMRI
from scipy.interpolate import RegularGridInterpolator

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments.unet import model_config as exp_config

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)
print('log_dir: ' + str(log_dir))
logging.info('Logging directory: %s' %log_dir)

# ==================================================================
# main function for inference
# ==================================================================
def run_inference():

    # ============================
    # log experiment details
    # ============================
    logging.info('============================================================')
    logging.info('EXPERIMENT NAME: %s' % exp_config.experiment_name)

    # ============================
    # Initialize step number - this is number of mini-batch runs
    # ============================
    init_step = 0

    # ============================
    # Find checkpoint file
    # ============================
    logging.info('============================================================')
    logging.info('Looking for checkpoint file')
    try:
        if os.path.exists('models/best_dice.ckpt'):
            init_checkpoint_path = os.path.join(log_dir, 'models/best_dice.ckpt')
            logging.info('============================================================')
            logging.info('Found best average dice checkpoint - restoring session from %s.' % init_checkpoint_path)
        else:
            init_checkpoint_path = utils.get_latest_model_checkpoint_path(log_dir, 'models/model.ckpt')
            logging.info('Checkpoint path: %s' % init_checkpoint_path)
            init_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1]) + 1  # plus 1 as otherwise starts with eval
            logging.info('Latest step was: %d' % init_step)
    except:
        logging.warning('Did not find init checkpoint. First need to run training successfully...')
        raise RuntimeError('No checkpoint to restore from available!')
    logging.info('============================================================')

    # ============================
    # Load data
    # ============================   
    # logging.info('============================================================')
    # logging.info('Loading training data from: ' + sys_config.project_data_root)
    # data_tr = data_freiburg_numpy_to_hdf5.load_data(basepath = sys_config.project_data_root,
    #                                                 idx_start = 0,
    #                                                 idx_end = 19,
    #                                                 train_test='train')
    # images_tr = data_tr['images_train']
    # labels_tr = data_tr['labels_train']
    # logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
    # logging.info('Shape of training labels: %s' %str(labels_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t]

    # Loading data from an hpc-predict-io MRI
    logging.info('============================================================')
    logging.info('Loading input FlowMRI from: ' + args.hpc_predict_input)
    flow_mri = FlowMRI.read_hdf5(args.hpc_predict_input)
    logging.info('============================================================')

    images_tr = np.concatenate([np.expand_dims(flow_mri.intensity,-1), flow_mri.velocity_mean], axis=-1).transpose([3,0,1,2,4])
    images_tr_preprocessed = np.zeros((images_tr.shape[0],*exp_config.image_size,images_tr.shape[-1]))
    # TODO: exact coordinate transformation

    np.array(np.meshgrid([0,1],[2,3],indexing='ij')).transpose([1,2,0])
    target_points = np.array(np.meshgrid(
                    np.linspace(flow_mri.geometry[0][0], flow_mri.geometry[0][-1], exp_config.image_size[0]),
                    np.linspace(flow_mri.geometry[1][0], flow_mri.geometry[1][-1], exp_config.image_size[1]),
                    np.linspace(flow_mri.geometry[2][0], flow_mri.geometry[2][-1], exp_config.image_size[2]),
        indexing='ij')).transpose([1,2,3,0])
    for t in range(images_tr.shape[0]):
        for v in range(images_tr.shape[-1]):
            rgi = RegularGridInterpolator(tuple(flow_mri.geometry), images_tr[t,...,v])
            images_tr_preprocessed[t,...,v] = rgi(target_points)
    images_tr = images_tr_preprocessed
    del images_tr_preprocessed

    logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
    images_tr_seg_logits = np.zeros(shape=(*images_tr.shape[:-1],exp_config.nlabels))

    logging.info('============================================================')
        
    # ================================================================
    # build the TF graph
    # ================================================================
    with tf.Graph().as_default():

        # ================================================================
        # create placeholders
        # ================================================================
        logging.info('Creating placeholders...')
        image_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size) + [exp_config.nchannels]
        # label_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size)
        images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name = 'images')
        # labels_pl = tf.placeholder(tf.uint8, shape=label_tensor_shape, name = 'labels')
        # learning_rate_pl = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
        training_pl = tf.placeholder(tf.bool, shape=[], name = 'training_or_testing')

        # ================================================================
        # Build the graph that computes predictions from the inference model
        # ================================================================
        logits = model.inference(images_pl,
                                 exp_config.model_handle,
                                 training_pl,
                                 exp_config)
        # seg_probs = tf.nn.softmax(logits)

        # # ================================================================
        # # Add ops for calculation of the training loss
        # # ================================================================
        # loss = model.loss(logits,
        #                   labels_pl,
        #                   exp_config.nlabels,
        #                   loss_type = exp_config.loss_type)
        # # Add the loss to tensorboard for visualizing its evolution as training proceeds
        # tf.summary.scalar('loss', loss)
        #
        # # ================================================================
        # # Add optimization ops
        # # ================================================================
        # train_op = model.training_step(loss,
        #                                exp_config.optimizer_handle,
        #                                learning_rate_pl)
        #
        # # ================================================================
        # # Add ops for model evaluation
        # # ================================================================
        # eval_loss = model.evaluation(logits,
        #                              labels_pl,
        #                              images_pl,
        #                              nlabels = exp_config.nlabels,
        #                              loss_type = exp_config.loss_type)

        # ================================================================
        # Build the summary Tensor based on the TF collection of Summaries.
        # ================================================================
        summary = tf.summary.merge_all()

        # ================================================================
        # Add init ops
        # ================================================================
        init_op = tf.global_variables_initializer()
        
        # ================================================================
        # Find if any vars are uninitialized
        # ================================================================
        logging.info('Adding the op to get a list of initialized variables...')
        uninit_vars = tf.report_uninitialized_variables()

        # ================================================================
        # create savers for each domain
        # ================================================================
        max_to_keep = 15
        saver = tf.train.Saver(max_to_keep = max_to_keep)
        # saver_best_dice = tf.train.Saver()

        # ================================================================
        # Create session
        # ================================================================
        sess = tf.Session()

        # ================================================================
        # create a summary writer
        # ================================================================
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        
        # # ================================================================
        # # summaries of the validation errors
        # # ================================================================
        # vl_error = tf.placeholder(tf.float32, shape=[], name='vl_error')
        # vl_error_summary = tf.summary.scalar('validation/loss', vl_error)
        # vl_dice = tf.placeholder(tf.float32, shape=[], name='vl_dice')
        # vl_dice_summary = tf.summary.scalar('validation/dice', vl_dice)
        # vl_summary = tf.summary.merge([vl_error_summary, vl_dice_summary])
        #
        # # ================================================================
        # # summaries of the training errors
        # # ================================================================
        # tr_error = tf.placeholder(tf.float32, shape=[], name='tr_error')
        # tr_error_summary = tf.summary.scalar('training/loss', tr_error)
        # tr_dice = tf.placeholder(tf.float32, shape=[], name='tr_dice')
        # tr_dice_summary = tf.summary.scalar('training/dice', tr_dice)
        # tr_summary = tf.summary.merge([tr_error_summary, tr_dice_summary])

        # ================================================================
        # freeze the graph before execution
        # ================================================================
        logging.info('Freezing the graph now!')
        tf.get_default_graph().finalize()

        # ================================================================
        # Run the Op to initialize the variables.
        # ================================================================
        logging.info('============================================================')
        logging.info('initializing all variables...')
        sess.run(init_op)
        
        # ================================================================
        # print names of uninitialized variables
        # ================================================================
        logging.info('============================================================')
        logging.info('This is the list of uninitialized variables:' )
        uninit_variables = sess.run(uninit_vars)
        for v in uninit_variables: print(v)

        # ================================================================
        # Restore session from a saved checkpoint
        # ================================================================
        # Restore session
        logging.info('============================================================')
        logging.info('No best dice checkpoint found - restoring session from: %s' % init_checkpoint_path)
        saver.restore(sess, init_checkpoint_path)

        # ================================================================
        # ================================================================        
        step = init_step
        # curr_lr = exp_config.learning_rate
        # best_dice = 0

        # ================================================================
        # run training epochs
        # ================================================================

        for batch_indices, batch in iterate_minibatches(images_tr, # TODO: modify for batch_size != 1
                                         batch_size = exp_config.batch_size):

            # curr_lr = exp_config.learning_rate
            start_time = time.time()
            x = batch

            # ===========================
            # run training iteration
            # ===========================
            feed_dict = {images_pl: x,
                         training_pl: False}
            seg_logits_values = sess.run([logits], feed_dict=feed_dict)
            images_tr_seg_logits[batch_indices, ...] = seg_logits_values[0][...]

            # ===========================
            # compute the time for this mini-batch computation
            # ===========================
            duration = time.time() - start_time

            # ===========================
            # write the summaries and print an overview fairly often
            # ===========================
            if (step+1) % exp_config.summary_writing_frequency == 0:
                logging.info('Step %d: (%.3f sec for the last step)' % (step+1, duration))

                # ===========================
                # update the events file
                # ===========================
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            step += 1
                
        sess.close()

        target_points = np.array(np.meshgrid(*flow_mri.geometry,
            indexing='ij')).transpose([1, 2, 3, 0])
        images_tr_seg_probs_mri = np.zeros((flow_mri.intensity.shape[3],*flow_mri.intensity.shape[:3]))
        images_tr_seg_logits_mri = np.zeros((*flow_mri.intensity.shape[:3], exp_config.nlabels))

        with tf.Graph().as_default():
            # ================================================================
            # freeze the graph before execution
            # ================================================================
            logging.info('Unfreezing the graph now!')
            tf.get_default_graph()._unsafe_unfinalize()

            logits_tensor_shape = list(flow_mri.intensity.shape[:3]) + [exp_config.nlabels]
            logits_pl = tf.placeholder(tf.float32, shape=logits_tensor_shape)
            seg_probs = tf.nn.softmax(logits_pl)

            with tf.Session() as sess:
                for t in range(images_tr_seg_logits.shape[0]):
                    for v in range(images_tr_seg_logits.shape[-1]):
                        rgi = RegularGridInterpolator(
                        ( np.linspace(flow_mri.geometry[0][0], flow_mri.geometry[0][-1], exp_config.image_size[0]),
                          np.linspace(flow_mri.geometry[1][0], flow_mri.geometry[1][-1], exp_config.image_size[1]),
                          np.linspace(flow_mri.geometry[2][0], flow_mri.geometry[2][-1], exp_config.image_size[2]) ),
                        images_tr_seg_logits[t, ..., v])
                        images_tr_seg_logits_mri[...,v] = rgi(target_points)
                    images_tr_seg_probs_mri[t, ...] = sess.run(seg_probs, feed_dict={logits_pl: images_tr_seg_logits_mri})[...,0]
            images_tr_seg_probs = images_tr_seg_probs_mri
            del images_tr_seg_probs_mri

        logging.info('============================================================')
        logging.info('Writing SegmentedFlowMRI to: ' + args.hpc_predict_output)
        segmented_flow_mri = SegmentedFlowMRI(
            flow_mri.geometry,
            flow_mri.time,
            flow_mri.time_heart_cycle_period,
            flow_mri.intensity,
            flow_mri.velocity_mean,
            flow_mri.velocity_cov,
            images_tr_seg_probs.transpose([1,2,3,0])[...])
        segmented_flow_mri.write_hdf5(args.hpc_predict_output)
        logging.info('============================================================')

# ==================================================================
# ==================================================================
def iterate_minibatches(images,
                        batch_size):
    '''
    Function to create mini batches from the dataset of a certain batch size 
    :param images: numpy dataset
    :param batch_size: batch size
    :return: mini batches
    '''

    # ===========================
    # generate indices to randomly select slices in each minibatch
    # ===========================
    n_images = images.shape[0]

    # ===========================
    # using only a fraction of the batches in each epoch
    # ===========================
    for b_i in range(0, n_images, batch_size):
        
        # HDF5 requires indices to be in increasing order
        batch_indices = np.arange(b_i,np.min([b_i+batch_size,n_images]))

        X = images[batch_indices, ...]

        yield batch_indices, X

