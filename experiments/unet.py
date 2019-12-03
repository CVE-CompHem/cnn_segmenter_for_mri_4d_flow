import model_zoo
import tensorflow as tf

from args import args

class ModelConfig:
    pass

model_config = ModelConfig()

import json
with open(args.model, 'r') as f:
    model_dict = json.load(f)
    for k,v in model_dict.items():
        setattr(model_config, k, v)

def rec_getattr(obj, name):
    names = name.split('.')
    if isinstance(obj, dict):
        ret = obj[names[0]]
    else:
        ret = getattr(obj, names[0])
    for k in names[1:]:
        ret = getattr(ret, k)
    return ret

model_config.model_handle = rec_getattr(model_zoo, model_config.model_handle)
model_config.optimizer_handle = rec_getattr(locals(), model_config.optimizer_handle)

# # ======================================================================
# # Model settings
# # ======================================================================
# model_handle = model_zoo.segmentation_cnn
# experiment_name = 'run3'
#
# # ======================================================================
# # data settings
# # ======================================================================
# data_mode = '3D'
# image_size = [144, 112, 48] # [x, y, time]
# nchannels = 4 # [intensity, vx, vy, vz]
# nlabels = 2 # [background, foreground]
#
# # ======================================================================
# # training settings
# # ======================================================================
# max_epochs = 1000
# batch_size = 8
# learning_rate = 1e-3
# optimizer_handle = tf.compat.v1.train.AdamOptimizer
# loss_type = 'dice'  # crossentropy/dice
# summary_writing_frequency = 20
# train_eval_frequency = 320
# val_eval_frequency = 320
# save_frequency = 800
#
# continue_run = False
# debug = True
# augment_data = True

# ======================================================================
# test settings
# ======================================================================
# iteration number to be loaded after training the model (setting this to zero will load the model with the best validation dice score)
#load_this_iter = 0
#batch_size_test = 4
#save_qualitative_results = True
#save_results_subscript = 'initial_training_unet'
#save_inference_result_subscript = 'inference_out_unet'