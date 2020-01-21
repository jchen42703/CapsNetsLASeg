from keras.models import Model
import six
from keras.layers import Input, Concatenate, Conv2D
from keras import backend as K
from functools import partial
from capsnets_laseg.models import model_utils
from capsnets_laseg.models.capsnets import CapsNetR3, CapsNetBasic

class AdaptiveUNet(model_utils.AdaptiveNetwork):
    """
    Isensee's 2D U-Net for Heart Segmentation from the MSD that follows the conditions:
        * pools until the feature maps axes are all of at <= 8
        * max # of pools = 6 for 2D
    Augmented to allow for use as a feature extractor
    Attributes:
        input_shape: The shape of the input including the number of input channels; (z, x, y, n_channels)
        n_convs: number of convolutions per module
        n_classes: number of output classes (default: 1, which is binary segmentation)
            * Make sure that it doesn't include the background class (0)
        max_pools: max number of max pooling layers
        starting_filters: number of filters at the highest depth
    """
    def __init__(self, input_shape, n_convs = 2, n_classes = 1, max_pools = 6, starting_filters = 30,):
        super().__init__(input_shape, max_pools, starting_filters, base_pool_size = 2)
        self.n_convs = n_convs
        self.n_classes = n_classes
        if self.ndim == 2:
            self.context_mod = partial(model_utils.context_module_2D, n_convs = n_convs)
            self.localization_mod = partial(model_utils.localization_module_2D, n_convs = n_convs)
        # automatically reassigns the max number of pools in a model (for cases where the actual max pools < inputted one)
        self.max_pools = max(self._pool_statistics())

    def build_model(self, include_top = False, input_layer = None, out_act = 'sigmoid'):
        """
        Returns a keras.models.Model instance.
        Args:
            include_top (boolean): Whether or not you want to have a segmentation layer
            input_layer: keras layer
                * if None, then defaults to a regular input layer based on the shape
            extractor: boolean on whether or not to use the U-Net as a feature extractor or not
            out_act: string representing the activation function for the last layer. Should be either "softmax" or "sigmoid".
            Defaults to "sigmoid".
        """
        if input_layer is None:
            input_layer = Input(shape = self.input_shape)
        skip_layers = []
        level = 0
        # context pathway (downsampling) [level 0 to (depth - 1)]
        while level < self.max_pools:
            if level == 0:
                skip, pool = self.context_mod(input_layer, self.filter_list[level], pool_size = self.pool_list[0])
            elif level > 0:
                skip, pool = self.context_mod(pool, self.filter_list[level], pool_size = self.pool_list[level])
            skip_layers.append(skip)
            level += 1
        convs_bottom = self.context_mod(pool, self.filter_list[level], pool_size = None) # No downsampling;  level at (depth) after the loop
        # localization pathway (upsampling with concatenation) [level (depth - 1) to level 1]
        while level > 0: # (** level = depth - 1 at the start of the loop)
            current_depth = level - 1
            if level == self.max_pools:
                upsamp = self.localization_mod(convs_bottom, skip_layers[current_depth], self.filter_list[current_depth],\
                                               upsampling_size = self.pool_list[current_depth])
            elif not level == self.max_pools:
                upsamp = self.localization_mod(upsamp, skip_layers[current_depth], self.filter_list[current_depth],\
                                               upsampling_size = self.pool_list[current_depth])
            level -= 1

        conv_seg = Conv2D(self.n_classes, kernel_size = (1,1), activation = out_act)(upsamp)
        # return feature maps
        if not include_top:
            extractor = Model(inputs = [input_layer], outputs = [upsamp])
            return extractor
        # return the segmentation
        elif include_top:
            unet = Model(inputs = [input_layer], outputs = [conv_seg])
            return unet

def SimpleUNet(include_top = False, input_layer = None, input_shape = (None, None, None), n_labels = 1, starting_filters = 32,
               depth = 4, n_convs = 2, activation = 'relu', padding = 'same', out_act = 'sigmoid'):
    """
    Builds a simple U-Net with batch normalization.
    Args:
        include_top: whether to include the final segmentation layer
        input_layer: keras layer
        input_shape: (x, y, n_channels)
        n_labels: number of labels (not including the background or `0` class). For example, a binary segmentation task
        would only have 1 label because it would only have 0 and 1 and 0 would be excluded.
        starting_filters: number of filters at the highest depth
        depth: Number of levels in the U-Net
        n_convs: number of convolutions in each level
        activation: activation function for each convolution
        padding: layer padding (default: 'same')
        out_act: activation function for last convolutional layer
    """
    # initializing some reusable components
    context_mod = partial(model_utils.context_module_2D, n_convs = n_convs, activation = activation)
    localization_mod = partial(model_utils.localization_module_2D, n_convs = n_convs, activation = activation, \
                               transposed_conv = True)
    filter_list = [starting_filters*(2**level) for level in range(0, depth)]
    pool_size = (2,2)
    max_pools = depth - 1
    # start of building the model
    if input_layer is None:
        input_layer = Input(shape = input_shape, name = 'x')
    skip_layers = []
    level = 0
    # context pathway (downsampling) [level 0 to (depth - 1)]
    while level < max_pools:
        if level == 0:
            skip, pool = context_mod(input_layer, filter_list[level], pool_size = pool_size)
        elif level > 0:
            skip, pool = context_mod(pool, filter_list[level], pool_size = pool_size)
        skip_layers.append(skip)
        level += 1
    convs_bottom = context_mod(pool, filter_list[level], pool_size = None) # No downsampling;  level at (depth) after the loop
    convs_bottom = context_mod(convs_bottom, filter_list[level], pool_size = None) # happens twice
    # localization pathway (upsampling with concatenation) [level (depth - 1) to level 1]
    while level > 0: # (** level = depth - 1 at the start of the loop)
        current_depth = level - 1
        if level == max_pools:
            upsamp = localization_mod(convs_bottom, skip_layers[current_depth], filter_list[current_depth],\
                                           upsampling_size = pool_size)
        elif not level == max_pools:
            upsamp = localization_mod(upsamp, skip_layers[current_depth], filter_list[current_depth],\
                                           upsampling_size = pool_size)
        level -= 1
    conv_transition = Conv2D(starting_filters, (1, 1), activation = activation)(upsamp)
    # return feature maps
    if not include_top:
        extractor = Model(inputs = [input_layer], outputs = [conv_transition])
        return extractor
    # return the segmentation
    elif include_top:
        # setting activation function based on the number of classes
        conv_seg = Conv2D(n_labels, (1,1), activation = out_act)(conv_transition)
        unet = Model(inputs = [input_layer], outputs = [conv_seg])
        return unet

class U_CapsNet(object):
    """
    The U-CapsNet architecture is made up of a U-Net feature extractor for a capsule network.
    Attributes:
        input_shape: sequence representing the input shape; (x, y, n_channels)
        n_class: the number of classes including the background class
        decoder: whether or not you want to include a reconstruction decoder in the architecture
    """
    def __init__(self, input_shape, n_class=2, decoder = True,):
        self.input_shape = input_shape
        self.n_class = n_class
        self.decoder = decoder

    def build_model(self, model_layer = None, capsnet_type = 'r3', upsamp_type = 'deconv'):
        """
        Builds the feature extractor + SegCaps network;
            Returns a keras.models.Model instance
        Args:
            model_layer: feature extractor
                * None: defaults to the AdaptiveUNet
                * 'simple': defaults to the basic U-Net
            capsnet_type: type of capsule network
                * 'r3':
                * 'basic':
            upsamp_type (str): one of ['deconv', 'subpix'] that represents the type of upsampling. Defaults to 'deconv'
        Returns:
            train_model: model for training
            eval_model: model for evaluation/inference
        """

        x = Input(self.input_shape, name = 'x')
        # initializing the U-Net feature extractor
        if model_layer is None:
            adap = AdaptiveUNet(2, self.input_shape, n_classes = self.n_class - 1, max_pools = 6, starting_filters = 5)
            model = adap.build_model(include_top = False, input_layer = x, out_act = 'sigmoid')
        elif model_layer == "simple":
            model = SimpleUNet(include_top = False, input_layer = x, input_shape = self.input_shape, out_act = 'sigmoid')
            # tensor_inp = simp_u.output
        else:
            model = model_layer
        # intializing the Capsule Network
        if capsnet_type.lower() == 'r3':
            train_model, eval_model = CapsNetR3(self.input_shape, n_class = 2, decoder = self.decoder, add_noise = False, \
                                      input_layer = model, upsamp_type = upsamp_type)
        elif capsnet_type.lower() == 'basic':
            train_model, eval_model = CapsNetBasic(self.input_shape, n_class = 2, decoder = self.decoder, add_noise = False, \
                                      input_layer = model)
        return train_model, eval_model
