import tensorflow as tf
import keras.backend as K
from keras import losses

def dice_soft(y_true, y_pred, smooth=1):
    """
    Calculates a differentiable Sorensen-Dice Coefficient.
    Args:
        y_true: labels
        y_pred: predicted logits
        smooth: value to avoid division by 0
    Returns:
        A float between 0 and 1 representing the dice coefficient
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coefficient_loss(y_true, y_pred):
    """
    Reduces the negative to be even more negative; works better for the unet
    Args:
        y_true: labels
        y_pred: predicted logits
    Returns:
        A negative float as the loss value.
    """
    return -dice_soft(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return 1-dice_soft(y_true, y_pred)

def dice_hard(y_true, y_pred, threshold=0.5, axis=[1,2,3], smooth=1e-5):
    """
    Non-differentiable Sørensen–Dice coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation i.e. labels are binary.
    Args:
        y_pred : tensor
            A distribution with shape: [batch_size, ....], (any dimensions).
        y_true : tensor
            A distribution with shape: [batch_size, ....], (any dimensions).
        threshold : float
            The threshold value to be true.
        axis : list of integer
            All dimensions are reduced, default ``[1,2,3]``.
        smooth : float
            This small value will be added to the numerator and denominator, see ``dice_coe``.
    """
    y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
    y_true = tf.cast(y_true > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(y_pred, y_true), axis=axis)
    l = tf.reduce_sum(y_pred, axis=axis)
    r = tf.reduce_sum(y_true, axis=axis)
    ## old axis=[0,1,2,3]
    # hard_dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # hard_dice = tf.clip_by_value(hard_dice, 0, 1.0-epsilon)
    ## new haodong
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice

def multi_class_dice(y_true, y_pred, smooth = 1e-5):
    """
    Simple multi-class dice coefficient that computes the average dice for each class.
    This implementation assumes a "channels_last" tensor format.
    Args:
        y_true:
        y_pred:
        smooth: small value to avoid division by 0
            default: 1e-5
    Returns:
        The mean dice coefficient over all of the classes.
    """
    n_dims = len(y_pred.shape) - 2 # subtracting the batch_size and n_channels dimensions
    axes = [axis for axis in range(n_dims + 1)] # to sum over all dimensions besides the channels (classes)
    intersect = K.sum(y_true * y_pred, axis=axes)
    numerator = 2 * intersect + smooth
    denominator = K.sum(y_true, axis = axes) + K.sum(y_pred, axis = axes) + smooth
    return K.mean(numerator / denominator)

def generalized_dice(y_true, y_pred, type_weight = "square", smooth = 1e-6):
    """
    The generalized dice coefficient as proposed by:
    * Sudre et al., 2017: https://arxiv.org/pdf/1707.03237.pdf
    This implementation assumes a "channels_last" tensor format.
    Args:
        y_true:
        y_pred:
        type_weight: mode representing the way the weights are calculated
            square: takes the reciprocal of the squared sum
            simple: takes the reciprocal of the sum
            uniform: weight = 1
        smooth: small value to avoid division by 0 (default value: 1e-6)
    """
    # Compute weights: "the contribution of each label is corrected by the inverse of its volume"
    n_dims = len(y_pred.shape) - 2 # subtracting the batch_size and n_channels dimensions
    axes = [axis for axis in range(n_dims + 1)] # to sum over all dimensions besides the channels (classes)
    ref_vol = K.sum(y_true, axis = 0)
    # prd_vol = K.sum(y_pred, axis = axes)
    if type_weight == 'square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    # Compute gen dice coef:
    intersect = K.sum(y_true * y_pred, axes)
    numerator = 2 * K.sum(weights * intersect)

    union = K.sum(y_true + y_pred, axes)
    denominator = K.sum(weights * union) + smooth
    # denominator = K.sum(weights * tf.maximum(ref_vol + pred_vol, 1))
    generalized_dice_score = numerator/denominator
    generalized_dice_score = tf.where(tf.is_nan(generalized_dice_score), 1.0,
                                      generalized_dice_score)
    return generalized_dice_score

def generalized_dice_loss(y_true, y_pred, type_weight = "square", smooth = 1e-6):
    return -generalized_dice(y_true, y_pred, type_weight = type_weight, smooth = smooth)

def dice_plus_xent_loss(y_true, y_pred, smooth = 1e-5):
    """
    Function to calculate the loss used in https://arxiv.org/pdf/1809.10486.pdf,
    no-new net, Isenseee et al (used to win the Medical Imaging Decathlon).
    It is the sum of the cross-entropy and the Dice-loss.
    Args:
        y_pred: the logits
        y_true: the one=hot encoded segmentation ground truth
    Return:
        the loss (cross_entropy + Dice)
    """
    y_pred = tf.cast(y_pred, tf.float32)
    multi_class = y_pred.shape[-1] >= 2
    if multi_class:
        loss_xent = K.mean(K.categorical_crossentropy(y_true, y_pred))
    elif not multi_class:
        assert y_pred.shape[-1] == 1, "Please check that your outputted segmenatations are single channel \
                                       for binary cross entropy."
        loss_xent = K.mean(K.binary_crossentropy(y_true, y_pred), axis = -1)
    # Dice as according to the paper:
    n_dims = len(y_pred.shape) - 2 # subtracting the batch_size and n_channels dimensions
    axes = [axis for axis in range(n_dims + 1)] # to sum over all dimensions besides the channels (classes)
    # axes = [0]
    numerator = 2.0 * K.sum(y_true * y_pred, axis = axes)
    denominator = K.sum(y_pred, axes) + K.sum(y_true, axes)
    if multi_class:
        loss_dice = K.mean((numerator + smooth) / (denominator + smooth))
    elif not multi_class:
        loss_dice = (numerator + smooth) / (denominator + smooth)
    return -loss_dice + loss_xent
