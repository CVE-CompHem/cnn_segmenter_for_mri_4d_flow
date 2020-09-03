# ==================================================================
# import 
# ==================================================================
import shutil
import logging
import os.path
import numpy as np
import tensorflow as tf
import utils
import model as model
import config.system as sys_config
import data_freiburg_numpy_to_hdf5

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import unet as exp_config

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)
logging.info('Logging directory: %s' %log_dir)

# ==================================================================
# main function for training
# ==================================================================
def run_training(continue_run):

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
    # if continue_run is set to True, load the model parameters saved earlier
    # else start training from scratch
    # ============================
    if continue_run:
        logging.info('============================================================')
        logging.info('Continuing previous run')
        try:
            init_checkpoint_path = utils.get_latest_model_checkpoint_path(log_dir, 'models/model.ckpt')
            logging.info('Checkpoint path: %s' % init_checkpoint_path)
            init_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1]) + 1  # plus 1 as otherwise starts with eval
            logging.info('Latest step was: %d' % init_step)
        except:
            logging.warning('Did not find init checkpoint. Maybe first run failed. Disabling continue mode...')
            continue_run = False
            init_step = 0
        logging.info('============================================================')

    # ============================
    # Load data
    # ============================   
    logging.info('============================================================')
    logging.info('Loading training data from: ' + sys_config.project_data_root)    
    data_tr = data_freiburg_numpy_to_hdf5.load_data(basepath = sys_config.project_data_root,
                                                    idx_start = 0,
                                                    idx_end = 19,
                                                    train_test='train')
    images_tr = data_tr['images_train']
    labels_tr = data_tr['labels_train']
    logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
    logging.info('Shape of training labels: %s' %str(labels_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t]

    logging.info('============================================================')
    logging.info('Loading validation data from: ' + sys_config.project_data_root)        
    data_vl = data_freiburg_numpy_to_hdf5.load_data(basepath = sys_config.project_data_root,
                                                    idx_start = 20,
                                                    idx_end = 24,
                                                    train_test='validation')
    images_vl = data_vl['images_validation']
    labels_vl = data_vl['labels_validation']
    logging.info('Shape of validation images: %s' %str(images_vl.shape))
    logging.info('Shape of validation labels: %s' %str(labels_vl.shape))
    logging.info('============================================================')

    # visualize some training images and their labels
    visualize_images = False
    if visualize_images is True:
        for sub_tr in range(20):
            utils.save_sample_image_and_labels_across_z(images_tr[sub_tr*32 : (sub_tr+1)*32, ...],
                                                        labels_tr[sub_tr*32 : (sub_tr+1)*32, ...],
                                                        log_dir + '/training_subject' + str(sub_tr))
            utils.save_sample_image_and_labels_across_t(images_tr[sub_tr*32 : (sub_tr+1)*32, ...],
                                                        labels_tr[sub_tr*32 : (sub_tr+1)*32, ...],
                                                        log_dir + '/training_subject' + str(sub_tr))
            
    # ================================================================
    # build the TF graph
    # ================================================================
    with tf.Graph().as_default():
        
        # ============================
        # set random seed for reproducibility
        # ============================
        tf.random.set_random_seed(exp_config.run_number)
        np.random.seed(exp_config.run_number)

        # ================================================================
        # create placeholders
        # ================================================================
        logging.info('Creating placeholders...')
        image_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size) + [exp_config.nchannels]
        label_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size)
        images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name = 'images')
        labels_pl = tf.placeholder(tf.uint8, shape=label_tensor_shape, name = 'labels')
        training_pl = tf.placeholder(tf.bool, shape=[], name = 'training_or_testing')

        # ================================================================
        # Build the graph that computes predictions from the inference model
        # ================================================================
        logits = model.inference(images_pl,
                                 exp_config.model_handle,
                                 training_pl)
        
        # ================================================================
        # Add ops for calculation of the training loss
        # ================================================================
        loss = model.loss(logits,
                          labels_pl,
                          exp_config.nlabels,
                          loss_type = exp_config.loss_type)
        
        # ================================================================
        # Add the loss to tensorboard for visualizing its evolution as training proceeds
        # ================================================================
        tf.summary.scalar('loss', loss)

        # ================================================================
        # Add optimization ops
        # ================================================================
        train_op = model.training_step(loss,
                                       exp_config.optimizer_handle,
                                       exp_config.learning_rate)

        # ================================================================
        # Add ops for model evaluation
        # ================================================================
        eval_loss = model.evaluation(logits,
                                     labels_pl,
                                     images_pl,
                                     nlabels = exp_config.nlabels,
                                     loss_type = exp_config.loss_type)

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
        saver_best_dice = tf.train.Saver()

        # ================================================================
        # Create session
        # ================================================================
        sess = tf.Session()

        # ================================================================
        # create a summary writer
        # ================================================================
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        
        # ================================================================
        # summaries of the validation errors
        # ================================================================
        vl_error = tf.placeholder(tf.float32, shape=[], name='vl_error')
        vl_error_summary = tf.summary.scalar('validation/loss', vl_error)
        vl_dice = tf.placeholder(tf.float32, shape=[], name='vl_dice')
        vl_dice_summary = tf.summary.scalar('validation/dice', vl_dice)
        vl_summary = tf.summary.merge([vl_error_summary, vl_dice_summary])
        
        # ================================================================
        # summaries of the training errors
        # ================================================================        
        tr_error = tf.placeholder(tf.float32, shape=[], name='tr_error')
        tr_error_summary = tf.summary.scalar('training/loss', tr_error)
        tr_dice = tf.placeholder(tf.float32, shape=[], name='tr_dice')
        tr_dice_summary = tf.summary.scalar('training/dice', tr_dice)
        tr_summary = tf.summary.merge([tr_error_summary, tr_dice_summary])
        
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
        # continue run from a saved checkpoint
        # ================================================================
        if continue_run:
            # Restore session
            logging.info('============================================================')
            logging.info('Restroring session from: %s' %init_checkpoint_path)
            saver.restore(sess, init_checkpoint_path)

        # ================================================================
        # ================================================================        
        step = init_step
        best_dice = 0

        # ================================================================
        # run training epochs
        # ================================================================
        while step < exp_config.max_steps:

            logging.info('============================================================')
            logging.info('Step %d' % step)
        
            for batch in iterate_minibatches(images_tr,
                                             labels_tr,
                                             batch_size = exp_config.batch_size):
                
                x, y = batch

                # ===========================
                # run training iteration
                # ===========================
                feed_dict = {images_pl: x,
                             labels_pl: y,
                             training_pl: True}                
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

                # ===========================
                # write the summaries and print an overview fairly often
                # ===========================
                if (step+1) % exp_config.summary_writing_frequency == 0:                    
                    logging.info('Step %d: loss = %.2f' % (step+1, loss_value))                                   
                    # ===========================
                    # update the events file
                    # ===========================
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # ===========================
                # compute the loss on the entire training set
                # ===========================
                if step % exp_config.train_eval_frequency == 0:

                    logging.info('Training Data Eval:')
                    [train_loss, train_dice] = do_eval(sess,
                                                       eval_loss,
                                                       images_pl,
                                                       labels_pl,
                                                       training_pl,
                                                       images_tr,
                                                       labels_tr,
                                                       exp_config.batch_size)                    

                    tr_summary_msg = sess.run(tr_summary, feed_dict={tr_error: train_loss, tr_dice: train_dice})
                    summary_writer.add_summary(tr_summary_msg, step)
                    
                # ===========================
                # save a checkpoint periodically
                # ===========================
                if step % exp_config.save_frequency == 0:

                    checkpoint_file = os.path.join(log_dir, 'models/model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)

                # ===========================
                # evaluate the model on the validation set
                # ===========================
                if step % exp_config.val_eval_frequency == 0:
                    
                    # ===========================
                    # Evaluate against the validation set
                    # ===========================
                    logging.info('Validation Data Eval:')
                    [val_loss, val_dice] = do_eval(sess,
                                                     eval_loss,
                                                     images_pl,
                                                     labels_pl,
                                                     training_pl,
                                                     images_vl,
                                                     labels_vl,
                                                     exp_config.batch_size)
                    
                    vl_summary_msg = sess.run(vl_summary, feed_dict={vl_error: val_loss, vl_dice: val_dice})
                    summary_writer.add_summary(vl_summary_msg, step)                    

                    # ===========================
                    # save model if the val dice is the best yet
                    # ===========================
                    if val_dice > best_dice:
                        best_dice = val_dice
                        best_file = os.path.join(log_dir, 'models/best_dice.ckpt')
                        saver_best_dice.save(sess, best_file, global_step=step)
                        logging.info('Found new average best dice on validation sets! - %f -  Saving model.' % val_dice)

                step += 1
                
        sess.close()

# ==================================================================
# ==================================================================
def do_eval(sess,
            eval_loss,
            images_placeholder,
            labels_placeholder,
            training_time_placeholder,
            volumes,
            labels,
            batch_size):

    '''
    Function for running the evaluations every X iterations on the training and validation sets. 
    :param sess: The current tf session 
    :param eval_loss: The placeholder containing the eval loss
    :param volumes_placeholder: Placeholder for the volumes
    :param labels_placeholder: Placeholder for the masks
    :param training_time_placeholder: Placeholder toggling the training/testing mode. 
    :param volumes: A numpy array or h5py dataset containing the volumes
    :param labels: A numpy array or h45py dataset containing the corresponding labels 
    :param batch_size: The batch_size to use. 
    :return: The average loss (as defined in the experiment), and the average dice over all `volumes`. 
    '''

    loss_ii = 0
    dice_ii = 0
    num_batches = 0

    for batch in iterate_minibatches(volumes, labels, batch_size=batch_size):
        
        x, y = batch
        if y.shape[0] < batch_size:
            continue

        feed_dict = {images_placeholder: x,
                     labels_placeholder: y,
                     training_time_placeholder: False}

        loss_batch, dice_batch = sess.run(eval_loss, feed_dict=feed_dict)
        loss_ii += loss_batch
        dice_ii += dice_batch
        num_batches += 1
        
    avg_loss = loss_ii / num_batches
    avg_dice = dice_ii / num_batches
    logging.info('  Average loss: %0.04f, average dice: %0.04f' % (avg_loss, avg_dice))

    return avg_loss, avg_dice

# ==================================================================
# ==================================================================
def iterate_minibatches(images,
                        labels,
                        batch_size):
    '''
    Function to create mini batches from the dataset of a certain batch size 
    :param images: numpy dataset
    :param labels: numpy dataset (same as images/volumes)
    :param batch_size: batch size
    :return: mini batches
    '''

    # ===========================
    # generate indices to randomly select slices in each minibatch
    # ===========================
    n_images = images.shape[0]
    random_indices = np.arange(n_images)
    np.random.shuffle(random_indices)
                    
    # ===========================
    # using only a fraction of the batches in each epoch
    # ===========================
    for b_i in range(0, n_images, batch_size):
        
        if b_i + batch_size > n_images:
            continue
        
        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])

        X = images[batch_indices, ...]
        y = labels[batch_indices, ...]
    
        # ===========================
        # augment the batch            
        # ===========================
        if exp_config.da_ratio > 0.0:
            X, y = utils.augment_data(X, y)
        
        yield X, y

# ==================================================================
# ==================================================================
def main():
    
    # ===========================
    # Create dir if it does not exist
    # ===========================
    continue_run = exp_config.continue_run
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
        tf.gfile.MakeDirs(log_dir + '/models')
        continue_run = False

    # ===========================
    # Copy experiment config file
    # ===========================
    shutil.copy(exp_config.__file__, log_dir)

    # ===========================
    # run training
    # ===========================
    run_training(continue_run)


# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()
