from skimage.transform import resize
import numpy as np
from builtins import range

def reshape(orig_img, append_value=-1024, new_shape=(512, 512, 512)):
    """
    Reshapes a numpy array with the specified values where the new shape must
    have >= to the number of voxels in the original image. If # of voxels in
    `new_shape` is > than the # of voxels in the original image, then the
    extra will be filled in by `append_value`.
    Args:
        orig_img:
        append_value: filler value
        new_shape: must have >= to the number of voxels in the original image.
    """
    reshaped_image = np.zeros(new_shape)
    reshaped_image[...] = append_value
    x_offset = 0
    y_offset = 0  # (new_shape[1] - orig_img.shape[1]) // 2
    z_offset = 0  # (new_shape[2] - orig_img.shape[2]) // 2

    reshaped_image[x_offset:orig_img.shape[0]+x_offset, y_offset:orig_img.shape[1]+y_offset, z_offset:orig_img.shape[2]+z_offset] = orig_img
    # insert temp_img.min() as background value

    return reshaped_image

def extract_nonint_region(image, mask=None, outside_value=0, coords=False):
    """
    Resizing image around a specified region (i.e. nonzero region)
    Args:
        image:
        mask: a segmentation labeled mask that is the same shaped as 'image'
            (optional; default: None)
        outside_value: (optional; default: 0)
        coords: boolean on whether or not to return boundaries
            (bounding box coords) (optional; default: False)
    Returns:
        the resized image
        segmentation mask (when mask is not None)
        a nested list of the mins and and maxes of each axis
        (when coords = True)
    """
    # Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ========================================================================
    # Changes: Added the ability to return the cropping coordinates
    pos_idx = np.where(image != outside_value)
    # fetching all of the min/maxes for each axes
    pos_x, pos_y, pos_z = pos_idx[1], pos_idx[2], pos_idx[0]
    minZidx, maxZidx = int(np.min(pos_z)), int(np.max(pos_z)) + 1
    minXidx, maxXidx = int(np.min(pos_x)), int(np.max(pos_x)) + 1
    minYidx, maxYidx = int(np.min(pos_y)), int(np.max(pos_y)) + 1
    # resize images
    resizer = (slice(minZidx, maxZidx), slice(minXidx, maxXidx), slice(minYidx, maxYidx))
    coord_list = [[minZidx, maxZidx], [minXidx, maxXidx], [minYidx, maxYidx]]
    # returns cropped outputs with the bbox coordinates
    if coords:
        if mask is not None:
            return (image[resizer], mask[resizer], coord_list)
        elif mask is None:
            return (image[resizer], coord_list)
    # returns just cropped outputs
    elif not coords:
        if mask is not None:
            return (image[resizer], mask[resizer])
        elif mask is None:
            return (image[resizer])

def resample_array(src_imgs, src_spacing, target_spacing, is_label=False):
    """
    Resamples a numpy array.
    Args:
        src_imgs: numpy array
        srs_spacing: list of the voxel spacing (must be the same dimension as
            the input numpy array)
        target_spacing: list of the target voxel spacings (must be the same
            dimension as the input numpy array)
    """
    # Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # =========================================================================
    # CHanges: Added the ability to resample groundtruth masks (using nearest neighbor)

    # Calcuating the target shape based on the target spacing
    src_spacing = np.round(src_spacing, decimals=3)
    target_shape = [int(src_imgs.shape[ix] * src_spacing[ix] / target_spacing[ix]) \
                    for ix in range(len(src_imgs.shape))]
    # Checking that none of the target shape is 0 or negative
    for i in range(len(target_shape)):
        try:
            assert target_shape[i] > 0
        except:
            raise AssertionError("AssertionError:", src_imgs.shape,
                                 src_spacing, target_spacing)

    img = src_imgs.astype(float)
    if is_label:
        # nearest neighbor interpolation for label
        resampled_img = resize(img, target_shape, order=0, clip=True,
                               mode='edge').astype('float32')
    elif not is_label:
        # third-order spline interpolation for image
        resampled_img = resize(img, target_shape, order=3, clip=True,
                               mode='edge').astype('float32')

    return resampled_img

def get_random_slice_idx(arr):
    slice_dim = arr.shape[0]
    return np.random.choice(slice_dim)

def get_positive_idx(label, channels_format="channels_last"):
    """
    Gets a random positive patch index that does not include the channels and
    batch_size dimensions.
    Args:
        label: one-hot encoded numpy array with the dims (n_channels, x,y,z)
    Returns:
        A numpy array representing a 3D random positive patch index
    """
    try:
        assert len(label.shape) == 4
    except AssertionError:
        # adds the channel dim if it doesn't already have it
        if channels_format == "channels_first":
            label = np.expand_dims(label, axis=0)
        elif channels_format == "channels_last":
            label = np.expand_dims(label, axis=-1)

    # "n_dims" numpy arrays of all possible positive pixel indices for the label
    if channels_format == "channels_first":
        pos_idx_new = np.nonzero(label)[1:]
    elif channels_format == "channels_last":
        pos_idx_new = np.nonzero(label)[:-1]

    # finding random positive class index
    pos_idx = np.dstack(pos_idx_new).squeeze()
    random_coord_idx = np.random.choice(pos_idx.shape[0]) # choosing random coords out of pos_idx
    random_pos_coord = pos_idx[random_coord_idx]
    return random_pos_coord

def clip_upper_lower_percentile(image, mask=None, percentile_lower=0.2,
                                percentile_upper=99.8):
    """
    Clipping values for positive class areas.
    Args:
        image:
        mask:
        percentile_lower:
        percentile_upper:
    Return:
        Image with clipped pixel intensities
    """
    # Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ===================================================================================================
    # Changes: Added the ability to have the function clip at only the necessary percentiles with no mask and removed the
    # automatic generation of a mask

    # finding the percentile values
    cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
    cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
    # clipping based on percentiles
    res = np.copy(image)
    if mask is not None:
        res[(res < cut_off_lower) & (mask !=0)] = cut_off_lower
        res[(res > cut_off_upper) & (mask !=0)] = cut_off_upper
    elif mask is None:
        res[(res < cut_off_lower)] = cut_off_lower
        res[(res > cut_off_upper)] = cut_off_upper
    return res

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
