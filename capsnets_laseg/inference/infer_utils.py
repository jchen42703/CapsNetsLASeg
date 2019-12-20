import numpy as np

def pad_nonint_extraction(image, orig_shape, coords, pad_value=0):
    """
    Pads the cropped output from `utils.shape_io`'s extract_nonint_region function
    Args:
        image: either the mask or the thresholded (= 0.5) segmentation
            prediction (x, y, z, n_channels)
        orig_shape: Original shape of the 3D volume including the channels
        coords: outputted coordinates from `extract_nonint_region`
    Returns:
        padded: numpy array of shape `orig_shape`
    """
    # trying to reverse the cropping with padding
    assert (len(coords) + 1) == len(orig_shape), \
        "Please make sure that orig_shape includes the channels dimension"
    padding = [[coords[i][0], orig_shape[i]-coords[i][1]]
                for i in range(len(orig_shape[:-1]))] + [[0,0]]
    padded = np.pad(image, padding, mode='constant', constant_values=pad_value)
    return padded

def undo_reshape_padding(image, orig_shape):
    """
    Undoes the padding done by the `reshape` function in `utils.shape_io`
    Args:
        image: numpy array that was reshaped to a larger size using reshape
        orig_shape: original shape before padding (do not include the channels)
    Returns:
        the reshaped image
    """
    return image[:orig_shape[0], :orig_shape[1], :orig_shape[2]]
