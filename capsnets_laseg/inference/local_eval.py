from keras_med_io.utils.shape_io import reshape
from keras_med_io.inference.infer_utils import pad_nonint_extraction, undo_reshape_padding
from capsnets_laseg.io.io_utils import isensee_preprocess, nii_to_np
from sklearn.metrics import precision_recall_fscore_support
import nibabel as nib
import numpy as np
import os

def pred_data_2D_per_sample(model, x_dir, y_dir, fnames, pad_shape, ct = False, batch_size = None):
    """
    Predicting 3D volumes slicewise (2D) one sample at a time.
    Assuming the input data is binary.
    Loads raw data, preprocesses it, predicts, and pads to original shape.
    Args:
        model: keras.models.Model instance
        x_dir: path to test images
        y_dir: path to the corresponding test masks
        fnames: files to evaluate in the directories; assuming that the input and labels are the same name
            * if it's None, we assume that it's all of the files in x_dir.
        pad_shape: of size (x,y); doesn't include batch size and channels
        batch_size: prediction batch size
    Returns:
        actual_y: actual labels
        padded_pred: stacked, thresholded predictions (padded to original shape)
    """
    if fnames is None:
        fnames = os.listdir(x_dir)

    y_list = []
    pred_list = []
    for id in fnames:
        # loads sample as a 3D numpy arr and then changes the type to float32
        x = nib.load(os.path.join(x_dir, id))
        y = nib.load(os.path.join(y_dir, id))

        # preprocessing
        preprocessed_x, preprocessed_y, coords = isensee_preprocess(x, y, orig_spacing = None, get_coords = True, ct = \
                                                                    ct, mean_patient_shape = (115, 320, 232))
        # pad to model input shape (predicting on a slicewise basis)
        _pad_shape = (preprocessed_x.shape[0],) + pad_shape
        reshaped_x = np.expand_dims(reshape(preprocessed_x, preprocessed_x.min(), new_shape = _pad_shape), -1)
        print("Predicting: ", id)
        predicted = model.predict(reshaped_x, batch_size = batch_size)
        # thresholding
        predicted[predicted >= 0.5] = 1
        predicted[predicted < 0.5] = 0
        # removing the reshape padding
        pred_no_pad = undo_reshape_padding(predicted, orig_shape = preprocessed_x.shape)
        # padding to original shape
        actual_label = nii_to_np(y)
        orig_shape = actual_label.shape + (1,)
        padded_pred = pad_nonint_extraction(pred_no_pad, orig_shape, coords, pad_value = 0)
        y_list.append(actual_label), pred_list.append(padded_pred)
    actual_y = np.vstack(y_list)
    padded_pred = np.vstack(pred_list)
    return actual_y, padded_pred

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
