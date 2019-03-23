from keras.layers import Add, Concatenate, LeakyReLU, BatchNormalization, \
                         Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
import numpy as np

class AdaptiveNetwork(object):
    """
    Adapts the network architecture to be based on the input dimensions
    * Computes the number of filters for each depth and pooling for each depth.
    * Primarily for making AdaptiveUNets recursively
        * Based on the configuration in the nnU-Net paper
    Attributes:
        input_shape: sequence of (x,y,z, n_channels)
        max_pools: max number of pools
        starting_filters: number of fitlers at the highest image resolution
        base_pool_size: either a int or tuple/list that represents the pool size. The pool size is inferred from the int.
    """
    def __init__(self, input_shape, max_pools = 5, starting_filters = 30, base_pool_size = 2):
        # self.n_convs = n_convs
        self.depth = max_pools + 1
        self.max_pools = max_pools
        self.input_shape = input_shape
        self.ndim = len(input_shape[:-1]) # not including channels dimension
        self.base_pool_size = base_pool_size
        if isinstance(self.base_pool_size, (tuple, list)):
            assert len(base_pool_size) == self.ndim, "Make sure that base_pool_size matches the number of dimensions!"
        elif isinstance(self.base_pool_size, int):
            self.base_pool_size = tuple([base_pool_size for i in range(self.ndim)])
        # computing the number of filters per depth and the pool sizes for each depth
        self.filter_list = [starting_filters*(2**level) for level in range(0, self.depth)]
        self.pool_list = self.generate_pool_sizes(self._pool_statistics())

    def _pool_statistics(self):
        """
        Finds the number of required pools for each axes.
        Returns:
            a list of the total number of pools for each axes (excluding the channels dimension)
            ** Note: can be used to return a corresponding list of pooling sizes; i.e. (1,2,2)
        """
        def check_shape(shape, shape_threshold):
            """
            Checks to see if any of the axes are <= shape_threshold;
            Args:
                shape:
                shape_threshold:
            Returns:
                bool_arr: A boolean array of all axes where:
                    True if # of feature maps >= 8
                    False if # of feature maps < 8
                lessmaxdown_idx: axes that are < 8
            """
            bool_arr = shape >= shape_threshold
            lessmaxdown_idx = np.where(bool_arr == False)[0] # returns all possible
            return (bool_arr, lessmaxdown_idx)

        def avoid_repeat_axes(dict_, idx):
            idx = int(idx)
            if idx in dict_.keys():
                pass
            elif idx not in dict_.keys():
                dict_[idx] = n_pool
            return dict_

        shape_temp = np.asarray(self.input_shape[:-1]) # no channels
        max_down_shape = np.asarray([8 for i in range(self.ndim)], dtype = np.int32) # max possible shape after downsampling to the deepest depth
        axes_dict = {}
        n_pool = 0
        divisor = np.asarray(self.base_pool_size) # pool by 2
        shape_temp_list = []
        while np.all(shape_temp > max_down_shape) or n_pool <= self.max_pools:
            # checks for <= 8 feature maps condition
            where_arr, append_idx = check_shape(shape_temp, max_down_shape)
            if append_idx.size > 0: # Adding the n_pools to a dict with the corresponding axes that have < 8 feature maps
                for idx in append_idx:
                    # Handles cases with repeat axes
                    axes_dict = avoid_repeat_axes(axes_dict, idx)
            if not (n_pool == self.max_pools): # so that division doesn't occur on the last iteration
                shape_temp = np.divide(shape_temp, divisor, where = where_arr)
            n_pool += 1
            shape_temp_list.append(shape_temp)
        axes_dict['min_shape'] = tuple(shape_temp.astype(np.int32))
        print("Deducing number of pools...\nMinimum shape: ", shape_temp)
        # deals with the case where after all pools are done, the output is all 8
#         if np.array_equal(axes_dict['min_shape'], max_down_shape):
        if (axes_dict['min_shape'] == tuple(max_down_shape)):
            axes_dict = {i: self.max_pools for i in range(self.ndim)}
        output_n_pools = tuple([axes_dict[i] for i in range(self.ndim)])

        print("Number of pools for each corresponding axis: ", output_n_pools)
        return output_n_pools

    def generate_pool_sizes(self, n_pools):
        """
        Generates pool sizes for a self.ndim array
        Args:
            n_pools: list of number of pools for axes with len 2 or 3
            base_pool_size: default pool size for MaxPooling and Upsampling layers
        """
        n_pools = np.asarray(n_pools)
        ordered = np.sort(n_pools)
        pool_list = [self.base_pool_size for pool in range(ordered[0])] # initial pool list until min pools
        # Iteratively augmenting and adding pool sizes to pool_list based on how many pools are left from the "previous" number
        for (first, before) in zip(range(1, n_pools.size), range(0, n_pools.size - 1)):
            if not ordered[first] == ordered[before]: # They shouldn't be the same, or else you wouldn't need to add more pools
                # augmenting the pool size
                new_size = self._get_single_pool_size(n_pools = n_pools, threshold = ordered[before])
                # Adding pool sizes up until ordered[first] pools
                for pool in range(ordered[first] - ordered[before]):
                    pool_list.append(new_size)
        assert len(pool_list) <= self.max_pools
        return pool_list

    def _get_single_pool_size(self, n_pools, threshold):
        """
        Handles cases where certain axes should not be pooled and returns an augmented pool size
        Args:
            n_pools:
            threshold: largest possible image size after all the pooling
        """
        pool_size = np.zeros((self.ndim,), dtype = np.int32)
        pool_coords = np.where(n_pools > threshold) # getting bool mask of axes that shouldn't be pooled
        no_pool_coords = np.where(n_pools <= threshold)
        pool_size[pool_coords] = 2
        pool_size[no_pool_coords] = 1
        return tuple(pool_size)

    @staticmethod
    def _compute_patch_shape(median_patient_shape):
        """
        Computes the patch shape from the median patient shape and the following conditions:
        1) if the median shape of the dataset is < 128, then use the median shape
           if the median shape of the dataset is > 128, then matches the Brain
           Tumour Segmentation patch/median image aspect ratio
        * Need to also adjust batch size but this was not implemented
        Args:
            median_patient_shape: the median shape of the dataset you're using
        """
        if not isinstance(median_patient_shape, np.ndarray):
            median_patient_shape = np.asarray(median_patient_shape)
        total_median_voxels = np.prod(median_patient_shape)
        #
        if total_median_voxels <= 128**3:
            patch_shape = tuple([np.median(median_patient_shape) for i in range(3)])
        elif total_median_voxels > 128**3:
            orig_patch_shape = np.asarray([128, 128, 128])
            orig_aspect_ratio = np.divide(orig_patch_shape, np.asarray([138, 169, 138]))
            patch_shape = np.round(orig_aspect_ratio * median_patient_shape, decimals = 0)
        return patch_shape

    def build_model(self):
        """
        Returns a keras.models.Model instance.
        """
        raise NotImplementedError

def context_module_2D(input_layer, n_filters, pool_size = (2,2), n_convs = 2, activation = 'LeakyReLU'):
    """
    [2D]; Channels_last
    Context module (Downsampling compartment of the U-Net): `n_convs` Convs (w/ LeakyReLU and BN) -> MaxPooling
    Args:
        input_layer:
        n_filters:
        pool_size: if None, there will be no pooling (default: (2,2))
        n_convs: Number of convolutions in the module
        activation: the activation name; the only advanced activation supported is 'LeakyReLU' (default: 'LeakyReLU')
    Returns:
        keras layer after double convs w/ LeakyReLU and BN in-between
        maxpooled output
    """
    for conv_idx in range(n_convs):
        if conv_idx == 0:
            if activation == 'LeakyReLU':
                conv = Conv2D(n_filters, kernel_size = (3,3), padding = 'same')(input_layer)
                act = LeakyReLU(0.3)(conv)
            elif not activation == 'LeakyReLU':
                act = Conv2D(n_filters, kernel_size = (3,3), padding = 'same', activation = activation)(input_layer)
            bn = BatchNormalization(axis = -1)(act)
        else:
            if activation == 'LeakyReLU':
                conv = Conv2D(n_filters, kernel_size = (3,3), padding = 'same')(bn)
                act = LeakyReLU(0.3)(conv)
            elif not activation == 'LeakyReLU':
                act = Conv2D(n_filters, kernel_size = (3,3), padding = 'same', activation = activation)(bn)
            bn = BatchNormalization(axis = -1)(act)
    if pool_size is not None:
        pool = MaxPooling2D(pool_size)(bn)
        return bn, pool
    elif pool_size is None:
        return bn

def localization_module_2D(input_layer, skip_layer, n_filters, upsampling_size = (2,2), n_convs = 2, activation = 'LeakyReLU', \
                           transposed_conv = False):
    """
    [2D]; Channels_last
    Localization module (Downsampling compartment of the U-Net): UpSampling3D -> `n_convs` Convs (w/ LeakyReLU and BN)
    Args:
        input_layer:
        skip_layer: layer with the corresponding skip connection (same depth)
        n_filters:
        upsampling_size:
        n_convs: Number of convolutions in the module
        activation: the activation name; the only advanced activation supported is 'LeakyReLU' (default: 'LeakyReLU')
        transposed_conv: boolean on whether you want transposed convs or UpSampling2D (default: False, which is UpSampling2D)
    Returns:
        keras layer after double convs w/ LeakyReLU and BN in-between
        upsampled output
    """
    if transposed_conv:
        upsamp = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding = 'same')(input_layer)
    elif not transposed_conv:
        upsamp = UpSampling2D(upsampling_size)(input_layer)
    concat = Concatenate(axis = -1)([upsamp, skip_layer])
    for conv_idx in range(n_convs):
        if conv_idx == 0:
            if activation == 'LeakyReLU':
                conv = Conv2D(n_filters, kernel_size = (3,3), padding = 'same')(concat)
                act = LeakyReLU(0.3)(conv)
            elif not activation == 'LeakyReLU':
                act = Conv2D(n_filters, kernel_size = (3,3), padding = 'same', activation = activation)(concat)
            bn = BatchNormalization(axis = -1)(act)
        else:
            if activation == 'LeakyReLU':
                conv = Conv2D(n_filters, kernel_size = (3,3), padding = 'same')(bn)
                act = LeakyReLU(0.3)(conv)
            elif not activation == 'LeakyReLU':
                act = Conv2D(n_filters, kernel_size = (3,3), padding = 'same', activation = activation)(bn)
            bn = BatchNormalization(axis = -1)(act)
    return bn
