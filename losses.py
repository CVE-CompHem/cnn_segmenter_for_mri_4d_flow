import tensorflow as tf

## ======================================================================
## ======================================================================
def compute_dice(logits, labels, epsilon=1e-10):
    '''
    Computes the dice score between logits and labels
    :param logits: Network output before softmax
    :param labels: ground truth label masks
    :param epsilon: A small constant to avoid division by 0
    :return: dice (per label, per image in the batch)
    '''

    with tf.name_scope('dice'):

        prediction = tf.nn.softmax(logits)
        intersection = tf.multiply(prediction, labels)
        
        reduction_axes = [1,2,3]        
        # compute area of intersection, area of GT, area of prediction (per image per label)
        tp = tf.reduce_sum(intersection, axis=reduction_axes) 
        tp_plus_fp = tf.reduce_sum(prediction, axis=reduction_axes) 
        tp_plus_fn = tf.reduce_sum(labels, axis=reduction_axes)

        # compute dice (per image per label)
        dice = 2 * tp / (tp_plus_fp + tp_plus_fn + epsilon)
        
        # =============================
        # if a certain label is missing in the GT of a certain image and also in the prediction,
        # dice[this_image,this_label] will be incorrectly computed as zero whereas it should be 1.
        # =============================
        
        # mean over all images in the batch and over all labels.
        mean_dice = tf.reduce_mean(dice)
        
        # mean over all images in the batch and over all foreground labels.
        mean_fg_dice = tf.reduce_mean(dice[:,1:])
        
    return dice, mean_dice, mean_fg_dice

## ======================================================================
## ======================================================================
def dice_loss(logits, labels):
    
    with tf.name_scope('dice_loss'):
        
        _, mean_dice, mean_fg_dice = compute_dice(logits, labels)
        
        loss = 1 - mean_dice

    return loss

## ======================================================================
## ======================================================================
def pixel_wise_cross_entropy_loss(logits, labels):

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    
    return loss