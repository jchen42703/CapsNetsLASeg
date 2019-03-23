from keras_med_io.utils.shape_io import reshape, resample_array, extract_nonint_region
from keras_med_io.utils.misc_utils import load_data
from keras_med_io.inference.infer_utils import *
import os

def evaluate_data_2D(model, x_dir, y_dir, fnames, pad_shape, ct = False):
    """
    Assuming the input data is binary.
    Loads raw data, preprocesses it, predicts, and pads to original shape.
    Args:
        x_dir: path to test images
        y_dir: path to the corresponding test masks
        fnames: files to evaluate in the directories; assuming that the input and labels are the same name
        pad_shape: of size (x,y); doesn't include batch size and channels
    Returns:
        padded_pred: prediction (padded to original shape)
        y_train: label
    """
    x_list = []
    y_list = []
    for id in fnames:
        # loads data as a numpy arr and then changes the type to float32
        x_train = load_data(os.path.join(x_dir, id))
        y_train = load_data(os.path.join(y_dir, id))

        preprocessed_x, preprocessed_y, coords = isensee_preprocess(x_train, y_train, orig_spacing = None, get_coords = True, ct = \
                                                                    ct, mean_patient_shape = (115, 320, 232))
        # pad to model input shape
        pad_shape = (preprocessed_x.shape[0],) + pad_shape + (1,)
        preprocessed_x = reshape(preprocessed_x, preprocessed_x.min(), new_shape = pad_shape)
        predicted = model.predict(preprocessed_x)
        # thresholding
        predicted[predicted >= 0.5] = 1
        predicted[predicted < 0.5] = 0
        # removing the reshape padding
        pred_no_pad = undo_reshape_padding(predicted, orig_shape = x_train.shape)
        # padding
        orig_shape = y_train.shape
        padded_pred = pad_nonint_extraction(pred_no_pad, orig_shape, coords)
        return padded_pred, y_train

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

def isensee_preprocess(input_image, mask, orig_spacing, get_coords = False, ct = False, mean_patient_shape = (115, 320, 232)):
    """
    Order:
    1) Cropping to non-zero regions
    2) Resampling to the median voxel spacing of the respective dataset
    3) Normalization

    Args:
        input_image:
        mask:
        orig_spacing: list/numpy array of voxel spacings corresponding to each axis of input_image and mask (assumes they have the same spacings)
        get_coords: boolean on whether to return extraction coords or not
        ct: boolean on whether `input_image` is a CT scan or not
        mean_patient_shape: obtained from Table 1. in the nnU-Net paper
    Returns:
        preprocessed input image and mask
    """
    # 1. Cropping
    if get_coords:
        extracted_img, extracted_mask, coords = extract_nonint_region(input_image, mask = mask, outside_value = 0, coords = True)
    elif not get_coords:
        extracted_img, extracted_mask = extract_nonint_region(input_image, mask = mask, outside_value = 0, coords = False)
    # 2. Resampling
    if orig_spacing is None: # renaming the variables because they don't get resampled
        resamp_img = extracted_img
        resamp_label = extracted_mask
    else:
        transposed_spacing = orig_spacing[::-1] # doing so because turning into numpy array moves the batch dimension to axis 0
        med_spacing = [np.median(transposed_spacing) for i in range(3)]
        resamp_img = resample_array(extracted_img, transposed_spacing, med_spacing, is_label = False)
        resamp_label = resample_array(extracted_mask, transposed_spacing, med_spacing, is_label = True)
    # 3. Normalization
    norm_img = zscore_isensee(resamp_img, ct = ct, mean_patient_shape = mean_patient_shape)
    if get_coords:
        return (norm_img, resamp_label, coords)
    elif not get_coords:
        return (norm_img, resamp_label)
