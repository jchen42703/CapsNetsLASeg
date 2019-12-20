import numpy as np
from skimage.transform import resize
from keras_med_io.utils.intensity_io import clip_upper_lower_percentile
from keras_med_io.utils.shape_io import extract_nonint_region, resample_array
from nibabel import Nifti1Image

def isensee_preprocess(input_image, mask, orig_spacing, get_coords=False,
                       ct=False, mean_patient_shape=(115, 320, 232)):
    """
    Order:
    1) Cropping to non-zero regions
    2) Resampling to the median voxel spacing of the respective dataset
    3) Normalization

    Args:
        input_image:
        mask:
        orig_spacing: list/numpy array of voxel spacings corresponding to each axis of input_image and mask (assumes they have the same spacings)
            * If it is left as None, then the images will not be resampled.
        get_coords: boolean on whether to return extraction coords or not
        ct: boolean on whether `input_image` is a CT scan or not
        mean_patient_shape: obtained from Table 1. in the nnU-Net paper
    Returns:
        preprocessed input image and mask
    """
    # converting types and axes order
    if isinstance(input_image, Nifti1Image):
        input_image = nii_to_np(input_image)
    if isinstance(mask, Nifti1Image):
        mask = nii_to_np(mask)
    # 1. Cropping
    if get_coords:
        extracted_img, extracted_mask, coords = extract_nonint_region(input_image,
                                                                      mask=mask,
                                                                      outside_value=0,
                                                                      coords=True)
    elif not get_coords:
        extracted_img, extracted_mask = extract_nonint_region(input_image,
                                                              mask=mask,
                                                              outside_value=0,
                                                              coords=False)
    # 2. Resampling
    if orig_spacing is None: # renaming the variables because they don't get resampled
        resamp_img = extracted_img
        resamp_label = extracted_mask
    else:
        transposed_spacing = orig_spacing[::-1] # doing so because turning into numpy array moves the batch dimension to axis 0
        med_spacing = [np.median(transposed_spacing) for i in range(3)]
        resamp_img = resample_array(extracted_img, transposed_spacing,
                                    med_spacing, is_label=False)
        resamp_label = resample_array(extracted_mask, transposed_spacing,
                                      med_spacing, is_label=True)
    # 3. Normalization
    norm_img = zscore_isensee(resamp_img, ct=ct,
                              mean_patient_shape=mean_patient_shape)
    if get_coords:
        return (norm_img, resamp_label, coords)
    elif not get_coords:
        return (norm_img, resamp_label)

# Normalization FN
def zscore_isensee(arr, ct, mean_patient_shape):
    """"
    Performs Z-Score normalization based on these conditions:
        CT:
            1) Clip to [0.5, 99.5] percentiles of intensity values
            (Paper does on whole training dataset, but this one does per img)
            2) Z-score norm on everything
        Other Modalities:
            1) Z-Score normalization individually
                * If # of voxels in crop < (mean # of voxels in orig / 4),
                normalization only on nonzero elements and everything else = 0
    Args:
        arr: cropped numpy array
        ct: boolean on whether or not the input image is a CT scan
        mean_patient_shape: list/tuple of the original input shape before cropping
    Return:
        A normalized numpy array according to the nnU-Net paper
    """
    cropped_voxels, mean_voxels = np.prod(np.array(arr.shape)), np.prod(np.array(mean_patient_shape))
    overcropped = cropped_voxels < (mean_voxels / 4)
    # CT normalization
    if ct:
        arr = clip_upper_lower_percentile(arr, percentile_lower=0.5,
                                          percentile_upper=99.5)
        return zscore_norm(arr)
    # Other modalities
    elif not ct:
        if overcropped:
            # only zscore norm on nonzero elements
            arr[arr != 0] = zscore_norm(arr[arr !=0])
        elif not overcropped:
            arr = zscore_norm(arr)
        return arr

def zscore_norm(arr):
    """
    Mean-Var Normalization
    * mean of 0 and standard deviation of 1
    Args:
        arr: numpy array
    Returns:
        A numpy array with a mean of 0 and a standard deviation of 1
    """
    shape = arr.shape
    arr = arr.flatten()
    norm_img = (arr-np.mean(arr)) / np.std(arr)
    return norm_img.reshape(shape)

def nii_to_np(nib_img):
    """
    Converts a 3D nifti image to a numpy array of (z, x, y) dims
    """
    return np.transpose(nib_img.get_fdata(), [-1, 0, 1])
