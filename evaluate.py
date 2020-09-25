# ========================================================
# import stuff
# ========================================================
import os
import logging
import numpy as np
import tensorflow as tf
import utils
import model as model
import config.system as sys_config
import data_freiburg_numpy_to_hdf5
import data_flownet
import sklearn.metrics as met

# ========================================================
# adding io interface developed by cscs
# Question: how to import from a directory several layers above the current one?
# For now, adding the path of the hpc-predict-io/python/ directory to sys.path
# ========================================================
import os, sys
current_dir_path = os.getcwd()
mr_io_dir_path = current_dir_path[:-39] + 'hpc-predict-io/python/'
sys.path.append(mr_io_dir_path)
from mr_io import FlowMRI, SegmentedFlowMRI

# ========================================================
# adding arg parser 
# ========================================================
from args import args

# ========================================================
# these tags decide how the predicted segmentation is saved.
# setting them directly here for now.
# later: read them from args.
# ========================================================
compute_quantitative_results = False
save_visual_results = False
save_segmentation = False

# ========================================================
# import exp_config
# ========================================================
# previously, I was taking in config vars as follows:
# from experiments import unet as exp_config
# this is the new way of taking in config vars
from experiments.unet import model_config as exp_config

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  

# ==================================================================
# set logging directory
# ==================================================================
if args.train is False and args.inference_output is not None:
    log_dir = args.inference_output
else:
    log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)

logging.info('Logging directory: %s' %log_dir)

# ============================================================================
# def get_segmentation
# ============================================================================
def get_segmentation(image):
        
    # ================================        
    # create an empty array to store the predicted segmentation
    # ================================            
    predicted_segmentation = np.zeros((image.shape[:-1]))
    
    # each subject has 32 zz slices and we want to do the prediction with batch size of 8
    for zz in range(32//8):
        predicted_segmentation[zz*8:(zz+1)*8,...] = sess.run(seg_mask,
                                                             feed_dict = {images_pl: image[zz*8:(zz+1)*8,...]})
        
    return predicted_segmentation
        
# ============================================================================
# Main inference function
# ============================================================================
def run_inference():

    # ===================================
    # go through each test subject
    # ===================================
    for n in range(1): #(1, 8):
        
        for r in [6]: # [6, 8, 10]:
            
            # ===================================
            # load test data
            # ===================================
            test_dataset = 'flownet' # freiburg / flownet
            subject_string = 'recon_R' + str(r) + '_volN' + str(n) + '_vn'
            
            # ===================================
            # for the freiburg dataset, all test images are loaded from the pre-made hdf5 file
            # ===================================
            if test_dataset is 'freiburg':
                logging.info('============================================================')
                logging.info('Loading test data from: ' + sys_config.project_data_root)        
                data = data_freiburg_numpy_to_hdf5.load_data(basepath = sys_config.project_data_root,
                                                             idx_start = 25,
                                                             idx_end = 28,
                                                             train_test='test')
                images = data['images_test']
                labels = data['labels_test']
                logging.info('Shape of test images: %s' %str(images.shape))
                logging.info('Shape of test labels: %s' %str(labels.shape))
                logging.info('============================================================')
                
            # ===================================
            # for the flownet dataset, the image data is read from the .mat files for the required subject and undersampling factor
            # ===================================
            else:
                logging.info('============================================================')
                test_data_path = '/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/data/eth_ibt/flownet/hannes/'
                test_data_path = test_data_path + subject_string + '.mat'
                logging.info('Loading test data from: ' + test_data_path)        
                images = data_flownet.read_image_mat(test_data_path)
                labels = np.zeros_like(images[...,0])
                logging.info('Shape of test images: %s' %str(images.shape))
                logging.info('Shape of test labels: %s' %str(labels.shape))
                logging.info('============================================================')

            # ===================================
            # Additionally, also load a data instance of the hpc-predict-io FlowMRI class
            # we can fill in the details of the particular image read just before, into the fields of the FlowMRI class instance
            # ===================================
            logging.info('============================================================')
            logging.info('Loading input FlowMRI from: ' + args.inference_input)
            flow_mri = FlowMRI.read_hdf5(args.inference_input)
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
                
                # ====================================
                # placeholders for images
                # ====================================    
                image_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size) + [exp_config.nchannels]
                images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name = 'images')
                
                # ====================================
                # create predict ops
                # ====================================        
                seg_logits, seg_prob, seg_mask = model.predict(images_pl, exp_config.model_handle)
                
                # ====================================
                # saver instance for loading the trained parameters
                # ====================================
                saver = tf.train.Saver()
                
                # ====================================
                # add initializer Ops
                # ====================================
                logging.info('Adding the op to initialize variables...')
                init_g = tf.global_variables_initializer()
                init_l = tf.local_variables_initializer()
        
                # ================================================================
                # Create session
                # ================================================================
                sess = tf.Session()    
                        
                # ====================================
                # Initialize
                # ====================================
                sess.run(init_g)
                sess.run(init_l)
        
                # ====================================
                # Restore trained segmentation model
                # ====================================
                path_to_model = log_dir 
                checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'models/best_dice.ckpt')
                logging.info('========================================================')
                logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
                saver.restore(sess, checkpoint_path)
                
                if test_dataset is 'freiburg':
                    subject_string = '000'

                # ================================   
                # open a text file for writing the mean dice scores for each subject that is evaluated
                # ================================  
                if compute_quantitative_results is True:
                    results_file = open(log_dir + '/results/' + test_dataset + '/' + subject_string + '.txt', "w")
                    results_file.write("================================== \n") 
                    results_file.write("Test results \n") 
                
                # ================================================================
                # For each test image, load the best model and compute the dice with this model
                # ================================================================
                dice_per_label_per_subject = []
                hsd_per_label_per_subject = []
                
                # ====================================
                # evaluate the test subjects, one at a time
                # ====================================
                num_subjects = images.shape[0] // 32
                for s in range(num_subjects):
        
                    logging.info('========================================================')            
                    logging.info('Predicting segmentation for test subject {}...'.format(s+1))
                    
                    if test_dataset is 'freiburg':
                        subject_string = 'subject_' + str(s+1)
                    
                    image_this_subject = images[s*32 : (s+1)*32, ...]
                    true_label_this_subject = labels[s*32 : (s+1)*32, ...]
                    
                    # predict segmentation
                    pred_label_this_subject = get_segmentation(image_this_subject)

                    # save segmentation
                    if save_segmentation is True:
                        logging.info('============================================================')
                        inference_output_file = os.path.join(args.inference_output,
                                                            os.path.basename(args.inference_input)[:-3] + '_cnn_segmented.h5')
                        logging.info('Writing SegmentedFlowMRI to: ' + inference_output_file)
                        segmented_flow_mri = SegmentedFlowMRI(flow_mri.geometry,
                                                              flow_mri.time,
                                                              flow_mri.time_heart_cycle_period,
                                                              flow_mri.intensity,
                                                              flow_mri.velocity_mean,
                                                              flow_mri.velocity_cov,
                                                              images_tr_seg_probs.transpose([1,2,3,0])[...])
                        segmented_flow_mri.write_hdf5(inference_output_file)
                        logging.info('============================================================')
                    
                    # save visual results
                    if save_visual_results is True:
                        utils.save_sample_results(im = image_this_subject,
                                                  pr = pred_label_this_subject,
                                                  gt = true_label_this_subject,
                                                  filename = log_dir + '/results/' + test_dataset + '/' + subject_string)
                    
                    # compute dice
                    if compute_quantitative_results is True:
                        dice_per_label_this_subject = met.f1_score(true_label_this_subject.flatten(),
                                                                pred_label_this_subject.flatten(),
                                                                average=None)
                        
                        # compute Hausforff distance 
                        hsd_per_label_this_subject = utils.compute_surface_distance(y1 = true_label_this_subject,
                                                                                    y2 = pred_label_this_subject,
                                                                                    nlabels = exp_config.nlabels)
                        
                        # write the mean fg dice of this subject to the text file
                        results_file.write("subject number" + str(s+1) + " :: dice (mean, std over all FG labels): ")
                        results_file.write(str(np.round(np.mean(dice_per_label_this_subject[1:]), 3)) + ", " + str(np.round(np.std(dice_per_label_this_subject[1:]), 3)))
                        results_file.write(", hausdorff distance (mean, std over all FG labels): ")
                        results_file.write(str(np.round(np.mean(hsd_per_label_this_subject), 3)) + ", " + str(np.round(np.std(dice_per_label_this_subject[1:]), 3)) + "\n")
                        
                        # append results to a list
                        dice_per_label_per_subject.append(dice_per_label_this_subject)
                        hsd_per_label_per_subject.append(hsd_per_label_this_subject)
                
                if compute_quantitative_results is True:
                    # ================================================================
                    # write per label statistics over all subjects    
                    # ================================================================
                    dice_per_label_per_subject = np.array(dice_per_label_per_subject)
                    hsd_per_label_per_subject =  np.array(hsd_per_label_per_subject)
                    
                    # ================================
                    # In the array images_dice, in the rows, there are subjects
                    # and in the columns, there are the dice scores for each label for a particular subject
                    # ================================
                    results_file.write("================================== \n") 
                    results_file.write("Label: dice mean, std. deviation over all subjects\n")
                    for i in range(dice_per_label_per_subject.shape[1]):
                        results_file.write(str(i) + ": " + str(np.round(np.mean(dice_per_label_per_subject[:,i]), 3)) + ", " + str(np.round(np.std(dice_per_label_per_subject[:,i]), 3)) + "\n")
                    
                    results_file.write("================================== \n") 
                    results_file.write("Label: hausdorff distance mean, std. deviation over all subjects\n")
                    for i in range(hsd_per_label_per_subject.shape[1]):
                        results_file.write(str(i+1) + ": " + str(np.round(np.mean(hsd_per_label_per_subject[:,i]), 3)) + ", " + str(np.round(np.std(hsd_per_label_per_subject[:,i]), 3)) + "\n")
                    
                    # ==================
                    # write the mean dice over all subjects and all labels
                    # ==================
                    results_file.write("================================== \n") 
                    results_file.write("DICE Mean, std. deviation over foreground labels over all subjects: " + str(np.round(np.mean(dice_per_label_per_subject[:,1:]), 3)) + ", " + str(np.round(np.std(dice_per_label_per_subject[:,1:]), 3)) + "\n")
                    results_file.write("HSD Mean, std. deviation over labels over all subjects: " + str(np.round(np.mean(hsd_per_label_per_subject), 3)) + ", " + str(np.round(np.std(hsd_per_label_per_subject), 3)) + "\n")
                    results_file.write("================================== \n") 
                    results_file.close()

# ==================================================================
# ==================================================================
def main():

    # ===========================
    # Run inference
    # ===========================
    run_inference()

# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()