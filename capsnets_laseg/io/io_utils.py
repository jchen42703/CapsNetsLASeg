import numpy as np
from skimage.transform import resize
from keras_med_io.utils.intensity_io import clip_upper_lower_percentile
# Normalization FN
def zscore_isensee(arr, ct, mean_patient_shape):
    """"
    Performs Z-Score normalization based on these conditions:
        CT:
            1) Clip to [0.5, 99.5] percentiles of intensity values (Paper does on whole training dataset, but this one does per img)
            2) Z-score norm on everything
        Other Modalities:
            1) Z-Score normalization individually
                * If # of voxels in crop < (mean # of voxels in orig / 4), normalization only on nonzero elements and everything else = 0
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
        arr = clip_upper_lower_percentile(arr, percentile_lower = 0.5, percentile_upper = 99.5)
        return zscore_norm(arr)
    # Other modalities
    elif not ct:
        if overcropped:
            arr[arr != 0] = zscore_norm(arr[arr !=0]) # only zscore norm on nonzero elements
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
