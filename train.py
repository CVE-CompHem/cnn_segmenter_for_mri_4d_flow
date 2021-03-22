# ==================================================================
# import modules
# ==================================================================
import shutil
import os.path
import utils
import h5py
import numpy as np
import model as model
import tensorflow as tf

# arguments
from args import args

# ==================================================================
# The config parameters are imported below
# This is done is a somewhat (and perhaps, unnecessarily complicated) manner!
# First, we look into the 'unet.py' file that is present inside the experiments directory
# This, in turn, reads the model parameters from args.model file, which, in turn, is set in the args.py file(!)
# Currently, the args.model is set to 'experiments/unet_neerav.json' file. 
# So, ultimately, the parameters that are read below are from the experiments/unet_neerav.json file.
# ==================================================================
from experiments.unet import model_config as exp_config

# ==================================================================
# import and setup logging
# ==================================================================
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ===================================
# parse arguments
# ===================================
# parser = argparse.ArgumentParser(prog = 'PROG')
# parser.add_argument('--training_data_filename', default = "full_path_and_filename_of_the_training_data")
# parser.add_argument('--training_output_dir', default = "directory_where_the_trained_model_should_be_stored")
# args = parser.parse_args()

# ===================================
# Set the logging directory
# ===================================
log_dir = args.training_output + exp_config.experiment_name
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
            init_step = 0 # plus 1 as otherwise starts with eval
        logging.info('============================================================')

    # ============================
    # Load training data
    # ============================   
    logging.info('============================================================')
    logging.info('Loading training data')    
    training_data_hdf5 = h5py.File(args.training_input, "r")
    images_tr = training_data_hdf5['images'] # e.g.[21, 112, 112, 20, 25, 4] : [num_subjects*num_r_values, nx, ny, nz, nt, 4]
    labels_tr = training_data_hdf5['labels'] # e.g.[21, 112, 112, 20, 25, 1] : [num_subjects*num_r_values, nx, ny, nz, nt, 1] # contains the prob. of the aorta (FG)
    logging.info('Shape of training images: %s' %str(images_tr.shape)) 
    logging.info('Shape of training labels: %s' %str(labels_tr.shape))
    logging.info('============================================================')
            
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
        image_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size) + [exp_config.nchannels] # e.g.[8, 112, 112, 25, 4] : [bs, nx, ny, nt, 4]
        label_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size) + [2] # # e.g.[8, 112, 112, 25, 2] : [bs, nx, ny, nt, 2]
        images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name = 'images')
        labels_pl = tf.placeholder(tf.float32, shape=label_tensor_shape, name = 'labels')
        training_pl = tf.placeholder(tf.bool, shape=[], name = 'training_or_testing')

        # ================================================================
        # Build the graph that computes predictions from the inference model
        # ================================================================
        logits = model.inference(images_pl,
                                 exp_config.model_handle,
                                 training_pl,
                                 exp_config.nlabels)
        
        # ================================================================
        # Add ops for calculation of the training loss
        # ================================================================
        loss = model.loss(logits,
                          labels_pl,
                          exp_config.nlabels,
                          loss_type = exp_config.loss_type,
                          labels_as_1hot = True)
        
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
                                     loss_type = exp_config.loss_type,
                                     labels_as_1hot = True)

        # ================================================================
        # Build the summary Tensor based on the TF collection of Summaries
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
        train_vars_list = []
        for v in tf.trainable_variables():
            train_vars_list.append(v)            
            print(v.name)
        saver = tf.train.Saver(var_list = train_vars_list)
        saver_best_dice = tf.train.Saver(var_list = train_vars_list)

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
        logging.info('=================================================')
        logging.info('initializing all variables...')
        sess.run(init_op)
        
        # ================================================================
        # print names of uninitialized variables
        # ================================================================
        logging.info('=================================================')
        logging.info('This is the list of uninitialized variables:' )
        uninit_variables = sess.run(uninit_vars)
        for v in uninit_variables: print(v)

        # ================================================================
        # continue run from a saved checkpoint
        # ================================================================
        if continue_run:
            # Restore session
            logging.info('=================================================')
            logging.info('Restroring session from: %s' %init_checkpoint_path)
            saver.restore(sess, init_checkpoint_path)

        # ================================================================
        # initialize counters
        # ================================================================        
        step = init_step
        best_dice = 0

        # ================================================================
        # run training epochs
        # ================================================================
        while step < exp_config.max_iterations:
        
            x, y = get_batch(images_tr, labels_tr, exp_config)

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
            if step % exp_config.summary_writing_frequency == 0:                    

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
                                                   exp_config)                    

                tr_summary_msg = sess.run(tr_summary, feed_dict={tr_error: train_loss, tr_dice: train_dice})
                summary_writer.add_summary(tr_summary_msg, step)
                
            # ===========================
            # save a checkpoint periodically
            # ===========================
            if step % exp_config.save_frequency == 0:

                logging.info('Step %d: saving checkpoint' % (step))
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
                                               images_tr,
                                               labels_tr,
                                               exp_config)
                
                vl_summary_msg = sess.run(vl_summary, feed_dict={vl_error: val_loss, vl_dice: val_dice})
                summary_writer.add_summary(vl_summary_msg, step)                    

                # ===========================
                # save model if the val dice is the best yet
                # ===========================
                if val_dice > best_dice:
                    best_dice = val_dice
                    best_file = os.path.join(log_dir, 'models/best_dice.ckpt')
                    saver_best_dice.save(sess, best_file)
                    # saver_best_dice.restore(sess, best_file)
                    logging.info('Found new average best dice on validation sets! - %f -  Saving model.' % val_dice)

            step += 1
                
        sess.close()
        
        # close the hdf5 file containing the training data
        training_data_hdf5.close()

# ==================================================================
# ==================================================================
def do_eval(sess,
            eval_loss,
            images_placeholder,
            labels_placeholder,
            training_time_placeholder,
            images,
            labels,
            exp_config):

    loss_ii = 0
    dice_ii = 0
    num_batches = 0

    for _ in range(10):
    
        x, y = get_batch(images, labels, exp_config)
        
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
def get_batch(images,
              labels,
              batch_size):
    '''
    Function to get a batch from the dataset
    :param images: numpy array
    :param labels: numpy array
    :param batch_size: batch size
    :return: batch
    '''

    x = np.zeros((exp_config.batch_size,
                  exp_config.image_size[0],
                  exp_config.image_size[1],
                  exp_config.image_size[2],
                  exp_config.nchannels), dtype = np.float32)
    
    y = np.zeros((exp_config.batch_size,
                  exp_config.image_size[0],
                  exp_config.image_size[1],
                  exp_config.image_size[2],
                  2), dtype = np.float32)
    
    for b in range(exp_config.batch_size):  
    
        # ===========================
        # generate indices to randomly select different x-y-t volumes in the batch
        # ===========================
        random_image_index = np.random.randint(images.shape[0])
        random_z_index = np.random.randint(images.shape[3])
        
        x[b, :, :, :, :] = images[random_image_index, :, :, random_z_index, :, :]
        y[b, :, :, :, 0] = 1 - labels[random_image_index, :, :, random_z_index, :, 0] # prob. of background
        y[b, :, :, :, 1] = labels[random_image_index, :, :, random_z_index, :, 0] # prob. of foreground

    # ===========================
    # augment the batch            
    # ===========================
    if exp_config.da_ratio > 0.0:
        x, y = utils.augment_data(x, y)
    
    return x, y

# ==================================================================
# ==================================================================
def train():
    
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
    shutil.copy(args.model, log_dir) # exp_config.__file__

    # ===========================
    # run training
    # ===========================
    run_training(continue_run)

# ==================================================================
# ==================================================================
def main():
    if args.debug_server is not None:
        try:
            import pydevd_pycharm
            debug_server_hostname, debug_server_port = args.debug_server.split(':')
            pydevd_pycharm.settrace(debug_server_hostname,
                                    port=int(debug_server_port), 
                                    stdoutToServer=True,
                                    stderrToServer=True)
        except:
            logging.error("Import error for pydevd_pycharm ignored (should not be running debug version).")

    # ===========================
    # Run the training
    # ===========================
    train()

# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()
