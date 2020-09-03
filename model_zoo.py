import tensorflow as tf
import logging
from experiments import unet as exp_config
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def segmentation_cnn(input_images, training): 
    
    with tf.name_scope('segmenter'):
        
        n0 = 8
                
        # ========================================================
        # ENCODER
        # ========================================================
        # ====================================
        # 1st Conv block - two conv layers, followed by max-pooling
        # Each conv layer consists of a convovlution, followed by batch normalization, followed by activation
        # ====================================
        conv1_1 = tf.layers.Conv3D(filters = n0, kernel_size=(3,3,3), padding='same')(input_images)
        conv1_1 = tf.layers.BatchNormalization()(conv1_1, training=training)
        conv1_1 = tf.nn.relu(conv1_1)
        conv1_2 = tf.layers.Conv3D(filters = n0, kernel_size=(3,3,3), padding='same')(conv1_1)
        conv1_2 = tf.layers.BatchNormalization()(conv1_2, training=training)
        conv1_2 = tf.nn.relu(conv1_2)
        maxpool1 = tf.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same')(conv1_2)

        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = tf.layers.Conv3D(filters = 2*n0, kernel_size=(3,3,3), padding='same')(maxpool1)
        conv2_1 = tf.layers.BatchNormalization()(conv2_1, training=training)
        conv2_1 = tf.nn.relu(conv2_1)
        conv2_2 = tf.layers.Conv3D(filters = 2*n0, kernel_size=(3,3,3), padding='same')(conv2_1)
        conv2_2 = tf.layers.BatchNormalization()(conv2_2, training=training)
        conv2_2 = tf.nn.relu(conv2_2)
        maxpool2 = tf.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same')(conv2_2)

        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = tf.layers.Conv3D(filters = 4*n0, kernel_size=(3,3,3), padding='same')(maxpool2)
        conv3_1 = tf.layers.BatchNormalization()(conv3_1, training=training)
        conv3_1 = tf.nn.relu(conv3_1)
        conv3_2 = tf.layers.Conv3D(filters = 4*n0, kernel_size=(3,3,3), padding='same')(conv3_1)
        conv3_2 = tf.layers.BatchNormalization()(conv3_2, training=training)
        conv3_2 = tf.nn.relu(conv3_2)
        maxpool3 = tf.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same')(conv3_2)

        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = tf.layers.Conv3D(filters = 8*n0, kernel_size=(3,3,3), padding='same')(maxpool3)
        conv4_1 = tf.layers.BatchNormalization()(conv4_1, training=training)
        conv4_1 = tf.nn.relu(conv4_1)
        conv4_2 = tf.layers.Conv3D(filters = 8*n0, kernel_size=(3,3,3), padding='same')(conv4_1)
        conv4_2 = tf.layers.BatchNormalization()(conv4_2, training=training)
        conv4_2 = tf.nn.relu(conv4_2)
        maxpool4 = tf.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same')(conv4_2)

        # ====================================
        # 5th Conv block / Bottleneck
        # ====================================
        conv5_1 = tf.layers.Conv3D(filters = 16*n0, kernel_size=(3,3,3), padding='same')(maxpool4)
        conv5_1 = tf.layers.BatchNormalization()(conv5_1, training=training)
        conv5_1 = tf.nn.relu(conv5_1)
        conv5_2 = tf.layers.Conv3D(filters = 16*n0, kernel_size=(3,3,3), padding='same')(conv5_1)
        conv5_2 = tf.layers.BatchNormalization()(conv5_2, training=training)
        conv5_2 = tf.nn.relu(conv5_2)

        # ========================================================
        # DECODER
        # ========================================================
        # ====================================
        # Upsampling 1, followed by two conv layers
        # Each conv layer consists of a convovlution, followed by batch normalization, followed by activation
        # ====================================
        upsample1 = tf.keras.layers.UpSampling3D(size=(2,2,2))(conv5_2)
        conc_up1 = tf.concat([conv4_2, upsample1], -1)
        conv6_1 = tf.layers.Conv3D(filters = 8*n0, kernel_size=(3,3,3), padding='same')(conc_up1)
        conv6_1 = tf.layers.BatchNormalization()(conv6_1, training=training)
        conv6_1 = tf.nn.relu(conv6_1)
        conv6_2 = tf.layers.Conv3D(filters = 8*n0, kernel_size=(3,3,3), padding='same')(conv6_1)
        conv6_2 = tf.layers.BatchNormalization()(conv6_2, training=training)
        conv6_2 = tf.nn.relu(conv6_2)
        
        # ====================================
        # Upsampling 2, followed by two conv layers
        # Each conv layer consists of a convovlution, followed by adaptive batch normalization, followed by activation
        # ====================================
        upsample2 = tf.keras.layers.UpSampling3D(size=(2,2,2))(conv6_2)
        conc_up2 = tf.concat([conv3_2, upsample2],-1)
        conv7_1 = tf.layers.Conv3D(filters = 4*n0, kernel_size=(3,3,3), padding='same')(conc_up2)
        conv7_1 = tf.layers.BatchNormalization()(conv7_1, training=training)
        conv7_1 = tf.nn.relu(conv7_1)
        conv7_2 = tf.layers.Conv3D(filters = 4*n0, kernel_size=(3,3,3), padding='same')(conv7_1)
        conv7_2 = tf.layers.BatchNormalization()(conv7_2, training=training)
        conv7_2 = tf.nn.relu(conv7_2)
        
        # ====================================
        # Upsampling 3, followed by two conv layers
        # Each conv layer consists of a convovlution, followed by adaptive batch normalization, followed by activation
        # ====================================
        upsample3 = tf.keras.layers.UpSampling3D(size=(2,2,2))(conv7_2)
        conc_up3 = tf.concat([conv2_2, upsample3],-1)
        conv8_1 = tf.layers.Conv3D(filters = 2*n0, kernel_size=(3,3,3), padding='same')(conc_up3)
        conv8_1 = tf.layers.BatchNormalization()(conv8_1, training=training)
        conv8_1 = tf.nn.relu(conv8_1)
        conv8_2 = tf.layers.Conv3D(filters = 2*n0, kernel_size=(3,3,3), padding='same')(conv8_1)
        conv8_2 = tf.layers.BatchNormalization()(conv8_2, training=training)
        conv8_2 = tf.nn.relu(conv8_2)
        
        # ====================================
        # Upsampling 4, followed by two conv layers
        # Each conv layer consists of a convovlution, followed by adaptive batch normalization, followed by activation
        # ====================================
        upsample4 = tf.keras.layers.UpSampling3D(size=(2,2,2))(conv8_2)
        conc_up4 = tf.concat([conv1_2, upsample4],-1)
        conv9_1 = tf.layers.Conv3D(filters = 1*n0, kernel_size=(3,3,3), padding='same')(conc_up4)
        conv9_1 = tf.layers.BatchNormalization()(conv9_1, training=training)
        conv9_1 = tf.nn.relu(conv9_1)
        conv9_2 = tf.layers.Conv3D(filters = 1*n0, kernel_size=(3,3,3), padding='same')(conv9_1)
        conv9_2 = tf.layers.BatchNormalization()(conv9_2, training=training)
        conv9_2 = tf.nn.relu(conv9_2)

        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        logits = tf.layers.Conv3D(filters=exp_config.nlabels, kernel_size=(3,3,3), padding='same')(conv9_2)
        
        # ====================================
        # print shapes at various layers in the network
        # ====================================
        logging.info('=======================================================')
        logging.info('Details of the segmentation CNN architecture')
        logging.info('=======================================================')
        logging.info('Shape of input: ' + str(input_images.shape))        
        logging.info('Shape after 1st max pooling layer: ' + str(maxpool1.shape))
        logging.info('Shape after 2nd max pooling layer: ' + str(maxpool2.shape))        
        logging.info('Shape after 3rd max pooling layer: ' + str(maxpool3.shape))        
        logging.info('Shape after 4th max pooling layer: ' + str(maxpool4.shape))            
        logging.info('=======================================================')
        logging.info('Before each maxpool, there are 2 conv blocks.')
        logging.info('Each conv block consists of conv3d (k=3), followed by BN, followed by relu.')
        logging.info('=======================================================')
        logging.info('Shape of the bottleneck layer: ' + str(conv5_2.shape))            
        logging.info('=======================================================')
        logging.info('Shape after 1st upsampling block: ' + str(conv6_2.shape))            
        logging.info('Shape after 2nd upsampling block: ' + str(conv7_2.shape))     
        logging.info('Shape after 3rd upsampling block: ' + str(conv8_2.shape))     
        logging.info('Shape after 4rd upsampling block: ' + str(conv9_2.shape)) 
        logging.info('=======================================================')
        logging.info('Each upsampling block consists of bilinear upsampling, followed by skip connection, followed by 2 conv blocks.')
        logging.info('=======================================================')
        logging.info('Shape of output (before softmax): ' + str(logits.shape)) 
        logging.info('=======================================================')

    return logits

## ======================================================================
## AE for RW aorta segmentation output correction
## ======================================================================
def conv_autoencoder(volumes, training): 
    
    with tf.name_scope('autoencoder'):
        # ========================================================
        # ENCODER
        # ========================================================
        
        # ====================================
        # 1st Conv block - two conv layers, followed by max-pooling
        # Each conv layer consists of a convovlution, followed by adaptive batch normalization, followed by activation
        # ====================================
        print('volumes shape'+str(volumes.shape))
        conv1_1 = tf.layers.Conv3D(filters=4, kernel_size=(3,3,3), padding='same')(volumes)
        conv1_1 = tf.layers.BatchNormalization()(conv1_1,training=training)
        conv1_1 = tf.nn.relu(conv1_1)
        conv1_2 = tf.layers.Conv3D(filters=4, kernel_size=(3,3,3), padding='same')(conv1_1)
        conv1_2 = tf.layers.BatchNormalization()(conv1_2,training=training)
        conv1_2 = tf.nn.relu(conv1_2)
        maxpool1 = tf.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same')(conv1_2)
        
        print('maxpool 1 shape'+str(maxpool1.shape))
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = tf.layers.Conv3D(filters=8, kernel_size=(3,3,3), padding='same')(maxpool1)
        conv2_1 = tf.layers.BatchNormalization()(conv2_1,training=training)
        conv2_1 = tf.nn.relu(conv2_1)
        conv2_2 = tf.layers.Conv3D(filters=8, kernel_size=(3,3,3), padding='same')(conv2_1)
        conv2_2 = tf.layers.BatchNormalization()(conv2_2,training=training)
        conv2_2 = tf.nn.relu(conv2_2)
        maxpool2 = tf.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same')(conv2_2)
        
        print('maxpool 2 shape'+str(maxpool2.shape))        
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = tf.layers.Conv3D(filters=16, kernel_size=(3,3,3), padding='same')(maxpool2)
        conv3_1 = tf.layers.BatchNormalization()(conv3_1,training=training)
        conv3_1 = tf.nn.relu(conv3_1)
        conv3_2 = tf.layers.Conv3D(filters=16, kernel_size=(3,3,3), padding='same')(conv3_1)
        conv3_2 = tf.layers.BatchNormalization()(conv3_2,training=training)
        conv3_2 = tf.nn.relu(conv3_2)
        maxpool3 = tf.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same')(conv3_2)
        
        print('maxpool 3 shape'+str(maxpool3.shape))  
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = tf.layers.Conv3D(filters=32, kernel_size=(3,3,3), padding='same')(maxpool3)
        conv4_1 = tf.layers.BatchNormalization()(conv4_1,training=training)
        conv4_1 = tf.nn.relu(conv4_1)
        conv4_2 = tf.layers.Conv3D(filters=32, kernel_size=(3,3,3), padding='same')(conv4_1)
        conv4_2 = tf.layers.BatchNormalization()(conv4_2,training=training)
        conv4_2 = tf.nn.relu(conv4_2)
        maxpool4 = tf.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same')(conv4_2)
        
        print('maxpool 4 shape'+str(maxpool4.shape))  
        # ====================================
        # FC layers of encoder
        # ====================================
        maxpool4_flat = tf.layers.Flatten()(maxpool4)
        print('maxpool 4 flat shape'+str(maxpool4_flat.shape))
        
        dense1 = tf.layers.Dense(units=1024, activation=tf.nn.relu)(maxpool4_flat)
        
        # ========================================================
        # BOTTLENECK / Latent space representation
        # ========================================================
        dense2 = tf.layers.Dense(units=256, activation=tf.nn.relu)(dense1)

        # ========================================================
        # DECODER
        # ========================================================
        # ====================================
        # FC layers of decoder
        # ====================================
        dense3 = tf.layers.Dense(units=1024, activation=tf.nn.relu)(dense2)
        dense4 = tf.layers.Dense(units=int(maxpool4_flat.shape[1]), activation=tf.nn.relu)(dense3)
        reshaped = tf.reshape(dense4,maxpool4.shape)

        # ====================================
        # Upsampling 1, followed by two conv layers
        # Each conv layer consists of a convovlution, followed by adaptive batch normalization, followed by activation
        # ====================================
        upsample1 = tf.keras.layers.UpSampling3D(size=(2,2,2))(reshaped)
        conv5_1 = tf.layers.Conv3D(filters=32, kernel_size=(3,3,3), padding='same')(upsample1)
        conv5_1 = tf.layers.BatchNormalization()(conv5_1,training=training)
        conv5_1 = tf.nn.relu(conv5_1)
        conv5_2 = tf.layers.Conv3D(filters=32, kernel_size=(3,3,3), padding='same')(conv5_1)
        conv5_2 = tf.layers.BatchNormalization()(conv5_2,training=training)
        conv5_2 = tf.nn.relu(conv5_2)
        print('upsample 1 shape'+str(conv5_2.shape))
        # ====================================
        # Upsampling 2, followed by two conv layers
        # Each conv layer consists of a convovlution, followed by adaptive batch normalization, followed by activation
        # ====================================
        upsample2 = tf.keras.layers.UpSampling3D(size=(2,2,2))(conv5_2)
        conv6_1 = tf.layers.Conv3D(filters=16, kernel_size=(3,3,3), padding='same')(upsample2)
        conv6_1 = tf.layers.BatchNormalization()(conv6_1,training=training)
        conv6_1 = tf.nn.relu(conv6_1)
        conv6_2 = tf.layers.Conv3D(filters=16, kernel_size=(3,3,3), padding='same')(conv6_1)
        conv6_2 = tf.layers.BatchNormalization()(conv6_2,training=training)
        conv6_2 = tf.nn.relu(conv6_2)
        print('upsample 2 shape'+str(conv6_2.shape))
        # ====================================
        # Upsampling 3, followed by two conv layers
        # Each conv layer consists of a convovlution, followed by adaptive batch normalization, followed by activation
        # ====================================
        upsample3 = tf.keras.layers.UpSampling3D(size=(2,2,2))(conv6_2)
        conv7_1 = tf.layers.Conv3D(filters=8, kernel_size=(3,3,3), padding='same')(upsample3)
        conv7_1 = tf.layers.BatchNormalization()(conv7_1,training=training)
        conv7_1 = tf.nn.relu(conv7_1)
        conv7_2 = tf.layers.Conv3D(filters=8, kernel_size=(3,3,3), padding='same')(conv7_1)
        conv7_2 = tf.layers.BatchNormalization()(conv7_2,training=training)
        conv7_2 = tf.nn.relu(conv7_2)
        print('upsample 3 shape'+str(conv7_2.shape))
        # ====================================
        # Upsampling 4, followed by two conv layers
        # Each conv layer consists of a convovlution, followed by adaptive batch normalization, followed by activation
        # ====================================
        upsample4 = tf.keras.layers.UpSampling3D(size=(2,2,2))(conv7_2)
        conv8_1 = tf.layers.Conv3D(filters=4, kernel_size=(3,3,3), padding='same')(upsample4)
        conv8_1 = tf.layers.BatchNormalization()(conv8_1,training=training)
        conv8_1 = tf.nn.relu(conv8_1)
        conv8_2 = tf.layers.Conv3D(filters=4, kernel_size=(3,3,3), padding='same')(conv8_1)
        conv8_2 = tf.layers.BatchNormalization()(conv8_2,training=training)
        conv8_2 = tf.nn.relu(conv8_2)
        print('upsample 4 shape'+str(conv8_2.shape))
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        logits = tf.layers.Conv3D(filters=exp_config.nlabels, kernel_size=(3,3,3), padding='same')(conv8_2)
        print('logits shape: '+str(logits.shape))

    return logits