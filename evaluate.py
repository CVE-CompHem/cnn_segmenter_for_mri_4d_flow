import os
import logging
import numpy as np
import tensorflow as tf
import utils
import model as model
import config.system as sys_config
import data_freiburg_numpy_to_hdf5
import data_flownet
from experiments import unet as exp_config
import sklearn.metrics as met

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')     

# ============================================================================
# def compute_and_save_results
# ============================================================================
def predict_segmentation(image):
        
    # ================================        
    # create an empty array to store the predicted segmentation
    # ================================            
    predicted_segmentation = np.zeros((image.shape[:-1]))
    
    # each subject has 32 zz slices and we want to do the prediction with batch size of 8
    for zz in range(32//8):
        predicted_segmentation[zz*8:(zz+1)*8,...] = sess.run(prediction,
                                                             feed_dict = {images_pl: image[zz*8:(zz+1)*8,...]})
        
    return predicted_segmentation
        
# ============================================================================
# Main function
# ============================================================================
if __name__ == '__main__':

    for n in range(1, 8):
        
        for r in [6, 8, 10]:
            
            # ===================================
            # load test data
            # ===================================
            test_dataset = 'flownet' # freiburg / flownet
            subject_string = 'recon_R' + str(r) + '_volN' + str(n) + '_vn'
            
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
                logits, softmax, prediction = model.predict(images_pl, exp_config.model_handle)
                
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
                # Restore training models
                # ====================================
                log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)
                path_to_model = log_dir 
                checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'models/best_dice.ckpt')
                logging.info('========================================================')
                logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
                saver.restore(sess, checkpoint_path)
            
                # ================================   
                # open a text file for writing the mean dice scores for each subject that is evaluated
                # ================================       
                if test_dataset is 'freiburg':
                    subject_string = '000'
                results_file = open(log_dir + '/results/' + test_dataset + '/' + subject_string + '.txt', "w")
                results_file.write("================================== \n") 
                results_file.write("Test results \n") 
                
                # ================================================================
                # For each test image, load the best model and compute the dice with this model
                # ================================================================
                dice_per_label_per_subject = []
                hsd_per_label_per_subject = []
                
                # ====================================
                # evaluate the test images 
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
                    pred_label_this_subject = predict_segmentation(image_this_subject)
                    
                    # save visual results
                    utils.save_sample_results(im = image_this_subject,
                                              pr = pred_label_this_subject,
                                              gt = true_label_this_subject,
                                              filename = log_dir + '/results/' + test_dataset + '/' + subject_string)
                    
                    # compute dice
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