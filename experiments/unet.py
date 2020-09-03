import model_zoo
import tensorflow as tf

# ======================================================================
# brief description of this experiment
# ======================================================================
# ...

# ======================================================================
# Model settings
# ======================================================================
model_handle = model_zoo.segmentation_cnn
run_number = 1
da_ratio = 0.0
experiment_name = 'unet3d_da_' + str(da_ratio) + '_r' + str(run_number)

# ======================================================================
# data settings
# ======================================================================
data_mode = '3D'
image_size = [144, 112, 48] # [x, y, time]
nchannels = 4 # [intensity, vx, vy, vz]
nlabels = 2 # [background, foreground]

# ======================================================================
# training settings
# ======================================================================
max_steps = 10000
batch_size = 8
learning_rate = 1e-3
optimizer_handle = tf.train.AdamOptimizer
loss_type = 'dice'  # crossentropy/dice
summary_writing_frequency = 20 # 4 times in each epoch (if n_tr_images=20, batch_size=8)
train_eval_frequency = 500 # every 4 epochs (if n_tr_images=20, batch_size=8)
val_eval_frequency = 500 # every 4 epochs (if n_tr_images=20, batch_size=8)
save_frequency = 1000 # every 10 epochs (if n_tr_images=20, batch_size=8)

continue_run = False
debug = True
augment_data = False

# ======================================================================
# test settings
# ======================================================================
# iteration number to be loaded after training the model (setting this to zero will load the model with the best validation dice score)
#load_this_iter = 0
#batch_size_test = 4
#save_qualitative_results = True
#save_results_subscript = 'initial_training_unet'
#save_inference_result_subscript = 'inference_out_unet'