from keras_med_io.utils.shape_io import reshape
from keras_med_io.inference.infer_utils import pad_nonint_extraction, undo_reshape_padding
from capsnets_laseg.io.io_utils import isensee_preprocess, nii_to_np
from sklearn.metrics import precision_recall_fscore_support
import nibabel as nib
import numpy as np
import os

def pred_data_2D_per_sample(model, x_dir, y_dir, fnames, pad_shape = (256, 320), batch_size = 2, \
                            mean_patient_shape = (115, 320, 232), ct = False):
    """
    Loads raw data, preprocesses it, predicts 3D volumes slicewise (2D) one sample at a time, and pads to the original shape.
    Assumptions:
        The segmentation task you're working with is binary.
        The multi-output models output only have two outputs: (prediction mask, reconstruction mask)
        The .nii.gz files have the shape: (x, y, z) where z is the number of slices.
    Args:
        model: keras.models.Model instance
        x_dir: path to test images
        y_dir: path to the corresponding test masks
        fnames: files to evaluate in the directories; assuming that the input and labels are the same name
            * if it's None, we assume that it's all of the files in x_dir.
        pad_shape: of size (x,y); doesn't include batch size and channels
        batch_size: prediction batch size
        mean_patient_shape: (z,x,y) representing the average shape. Defaults to (115, 320, 232).
        ct: whether or not the data is a CT scan or not. Defaults to False.
    Returns:
        actual_y: actual labels
        padded_pred: stacked, thresholded predictions (padded to original shape)
        padded_recon: stacked, properly padded reconstruction. Defaults to None if the model only outputs segmentations.
        orig_images: original input images
    """
    # can automatically infer the filenames to use (ensure that there are no junk files in x_dir)
    if fnames is None:
        fnames = os.listdir(x_dir)
    # lists that hold the arrays
    y_list = []
    pred_list = []
    recon_list = []
    orig_list = []
    for id in fnames:
        # loads sample as a 3D numpy arr and then changes the type to float32
        x = nib.load(os.path.join(x_dir, id))
        y = nib.load(os.path.join(y_dir, id))
        orig_images, actual_label = nii_to_np(x), nii_to_np(y) # purpose is to transpose axes to (z,x, y)
        orig_shape = orig_images.shape + (1,)
        # preprocessing
        preprocessed_x, preprocessed_y, coords = isensee_preprocess(x, y, orig_spacing = None, get_coords = True, ct = \
                                                                    ct, mean_patient_shape = mean_patient_shape)
        # pad to model input shape (predicting on a slicewise basis)
        _pad_shape = (preprocessed_x.shape[0],) + pad_shape # unique to each volume because the n_slice varies
        # preparing the shape for the model (reshaping to model input shape and adding a channel dimension)
        reshaped_x = np.expand_dims(reshape(preprocessed_x, preprocessed_x.min(), new_shape = _pad_shape), -1)
        # prediction
        print("Predicting: ", id)
        predicted = model.predict(reshaped_x, batch_size = batch_size)
        # inferring that the model has a reconstruction decoder based on the outputted predictions
        if isinstance(predicted, (list, tuple)):
            predicted, reconstruction = predicted
            # properly converting the reconstruction to the original shape
            padded_recon = undo_reshape_and_nonint_extraction(reconstruction, prior_reshape_shape = preprocessed_x.shape, \
                                                             orig_shape = orig_shape, coords = coords, pad_value = 0)
            recon_list.append(padded_recon)
        # thresholding
        predicted[predicted >= 0.5] = 1
        predicted[predicted < 0.5] = 0
        # properly converting the prediction mask to the original shape
        padded_pred = undo_reshape_and_nonint_extraction(predicted, prior_reshape_shape = preprocessed_x.shape, \
                                                         orig_shape = orig_shape, coords = coords, pad_value = 0)
        y_list.append(actual_label), pred_list.append(padded_pred), orig_list.append(orig_images)
    # stacking the lists
    actual_y, padded_pred, orig_images = np.vstack(y_list), np.vstack(pred_list), np.vstack(orig_list)
    try:
        padded_recon = np.vstack(recon_list)
    except ValueError: # can't stack empty list
        padded_recon = None
    return (actual_y, padded_pred, padded_recon, orig_images)

def undo_reshape_and_nonint_extraction(pred, prior_reshape_shape, orig_shape, coords, pad_value = 0):
    """
    Undoes the reshape padding and pads to compensate for `extract_nonint_region`.
    Args:
        pred: prediction from model in the form (z, x, y)
        prior_reshape_shape: the shape after `isensee_preprocess` and before it was reshaped to be fed into a model
        orig_shape: original shape of the raw Nifti1Image, with axes transposed to (z, x, y)
        coords: Coordinates from `extract_nonint_region`
        pad_value: Pad value for `pad_nonint_extraction`
    Returns:
        The padded prediction that corresponds to the original labels except with the axes transposed to (z, x, y)
    """
    # removing the reshape padding
    pred_no_pad = undo_reshape_padding(pred, orig_shape = prior_reshape_shape)
    # padding to original shape
    padded_pred = pad_nonint_extraction(pred_no_pad, orig_shape, coords, pad_value = 0)
    return padded_pred

def evaluate_2D(y, pred):
    """
    Evaluates the generated predictions (binary only) with precision, recall and dice (f1-score)
    Args:
        y: binary groundtruth mask
        pred: thresholded binary segmentation mask
    Returns:
        (precision, recall, dice)
    """
    precision, recall, dice, _ = precision_recall_fscore_support(y.flatten(), pred.flatten(), average = 'binary')
    print("Precision: ", precision, "\nRecall: ", recall, "\nDice: ", dice)
    return (precision, recall, dice)

def load_raw_data(x_dir, y_dir, fnames):
    """
    Args:
        x_dir: path to test images
        y_dir: path to the corresponding test masks
        fnames: files to evaluate in the directories; assuming that the input and labels are the same name
    Returns:
        x_list: list of the raw numpy arrays
        y_list: list of the raw corresponding numpy arrays
    """
    x_list = []
    y_list = []
    for id in fnames:
        # loads data as a numpy arr and then changes the type to float32
        x_train = load_data(os.path.join(x_dir, id))
        y_train = load_data(os.path.join(y_dir, id))
        x_list.append(x_train), y_list.append(y_train)
    return x_list, y_list
