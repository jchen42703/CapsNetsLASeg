from keras import layers, models
from keras import backend as K
K.set_image_data_format('channels_last')

from .caps_layers import ConvCapsuleLayer, DeconvCapsuleLayer, Mask, Length

def CapsNetR3(input_shape, n_class=2, decoder=True, add_noise=False,
              input_layer=None, upsamp_type='deconv'):
    """
    Args:
        input_shape: Must be of shape (x, y, 1)
        n_class: number of classes (includes the background class)
        decoder: boolean on whether to include a decoder in the models
        add_noise: boolean on whether to have a layer that adds noise
        input_layer: keras layer; if it is None, then we automatically
            initialize one from `input_shape`
        (Also supports passing a model to serve as a feature extractor)
        upsamp_type (str): one of ['deconv', 'subpix'] that represents the
            type of upsampling. Defaults to 'deconv'
    Returns:
        train_model: two inputs (x,y) for training
        eval_model: one input (x) for evaluation/inference
        manipulate_model: (if add_noise = True)
    """
    if input_layer is None:
        input_layer = layers.Input(shape=input_shape, name='x')
        x = input_layer
    elif isinstance(input_layer, models.Model):
        x = input_layer.inputs[0]
        input_layer = input_layer.output

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=16, kernel_size=5, strides=1,
                          padding='same', activation='relu',
                          name='conv1')(input_layer)

    # Reshape layer to be 1 capsule x [filters] atoms
    _, H, W, C = conv1.get_shape()
    conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=2,
                                    num_atoms=16, strides=2, padding='same',
                                    routings=1, name='primarycaps')(conv1_reshaped)

    # Layer 2: Convolutional Capsule
    conv_cap_2_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=4,
                                    num_atoms=16, strides=1, padding='same',
                                    routings=3, name='conv_cap_2_1')(primary_caps)

    # Layer 2: Convolutional Capsule
    conv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=32,
                                    strides=2, padding='same', routings=3,
                                    name='conv_cap_2_2')(conv_cap_2_1)

    # Layer 3: Convolutional Capsule
    conv_cap_3_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32,
                                    strides=1, padding='same', routings=3,
                                    name='conv_cap_3_1')(conv_cap_2_2)

    # Layer 3: Convolutional Capsule
    conv_cap_3_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=64,
                                    strides=2, padding='same', routings=3,
                                    name='conv_cap_3_2')(conv_cap_3_1)

    # Layer 4: Convolutional Capsule
    conv_cap_4_1 = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32,
                                    strides=1, padding='same', routings=3,
                                    name='conv_cap_4_1')(conv_cap_3_2)

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=8,
                                        num_atoms=32, upsamp_type=upsamp_type,
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_1_1')(conv_cap_4_1)

    # Skip connection
    up_1 = layers.Concatenate(axis=-2, name='up_1')([deconv_cap_1_1, conv_cap_3_1])

    # Layer 1 Up: Deconvolutional Capsule
    deconv_cap_1_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4,
                                      num_atoms=32, strides=1, padding='same',
                                      routings=3, name='deconv_cap_1_2')(up_1)

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=4,
                                        num_atoms=16, upsamp_type=upsamp_type,
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_2_1')(deconv_cap_1_2)

    # Skip connection
    up_2 = layers.Concatenate(axis=-2, name='up_2')([deconv_cap_2_1, conv_cap_2_1])

    # Layer 2 Up: Deconvolutional Capsule
    deconv_cap_2_2 = ConvCapsuleLayer(kernel_size=5, num_capsule=4,
                                      num_atoms=16, strides=1, padding='same',
                                      routings=3, name='deconv_cap_2_2')(up_2)

    # Layer 3 Up: Deconvolutional Capsule
    deconv_cap_3_1 = DeconvCapsuleLayer(kernel_size=4, num_capsule=2,
                                        num_atoms=16, upsamp_type=upsamp_type,
                                        scaling=2, padding='same', routings=3,
                                        name='deconv_cap_3_1')(deconv_cap_2_2)

    # Skip connection
    up_3 = layers.Concatenate(axis=-2, name='up_3')([deconv_cap_3_1, conv1_reshaped])

    # Layer 4: Convolutional Capsule: 1x1
    seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16,
                                strides=1, padding='same', routings=3,
                                name='seg_caps')(up_3)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(seg_caps)
    _, H, W, C, A = seg_caps.get_shape()

    if decoder:
        # Decoder network.
        prediction_shape = input_shape[:-1]+(1,) # assumed

        y = layers.Input(shape=prediction_shape, name='y')
        masked_by_y = Mask()([seg_caps, y])  # The true label is used to mask the output of capsule layer. For training
        masked = Mask()(seg_caps)  # Mask using the capsule with maximal length. For prediction

        def shared_decoder(mask_layer):
            reshaped_ = (H.value, W.value, C.value * A.value)
            recon_remove_dim = layers.Reshape(reshaped_)(mask_layer)

            recon_1 = layers.Conv2D(filters=64, kernel_size=1, padding='same',
                                    kernel_initializer='he_normal',
                                    activation='relu', name='recon_1')(recon_remove_dim)

            recon_2 = layers.Conv2D(filters=128, kernel_size=1, padding='same',
                                    kernel_initializer='he_normal',
                                    activation='relu', name='recon_2')(recon_1)

            out_recon = layers.Conv2D(filters=1, kernel_size=1, padding='same',
                                      kernel_initializer='he_normal',
                                      activation='sigmoid', name='out_recon')(recon_2)

            return out_recon

        # Models for training and evaluation (prediction)
        train_model = models.Model(inputs=[x, y],
                                   outputs=[out_seg, shared_decoder(masked_by_y)])
        eval_model = models.Model(inputs=x,
                                  outputs=[out_seg, shared_decoder(masked)])

    elif not decoder:
        # No decoder network (single input)
        train_model = models.Model(inputs=x, outputs=out_seg)
        eval_model = models.Model(inputs=x, outputs=out_seg)

    if add_noise:
        # manipulate model: Adding noise to the prediction and masks
        noise = layers.Input(shape=seg_caps.get_shape().as_list()[1:], name='noise')
        noised_seg_caps = layers.Add()([seg_caps, noise])
        masked_noised_y = Mask()([noised_seg_caps, y])
        if decoder: # (x,y)
            manipulate_model = models.Model(inputs=[x, y, noise],
                                            outputs=shared_decoder(masked_noised_y))
        elif not decoder: # (x,)
            manipulate_model = models.Model(inputs=[x, noise],
                                            outputs=shared_decoder(masked))
        return (train_model, eval_model, manipulate_model)
    else:
        return (train_model, eval_model)

def CapsNetBasic(input_shape, n_class=2, decoder=True, add_noise=False,
                 input_layer=None):
    """
    Args:
        input_shape: Must be of shape (x, y, 1)
        n_class: number of classes (includes the background class)
        decoder: boolean on whether to include a decoder in the models
        add_noise: boolean on whether to have a layer that adds noise
        input_layer: keras layer; if it is None, then we automatically
            initialize one from `input_shape`
    Returns:
        train_model: two inputs (x,y) for training
        eval_model: one input (x) for evaluation/inference
        manipulate_model: (if add_noise = True)
    """
    if input_layer is None:
        input_layer = layers.Input(shape=input_shape, name='x')
        x = input_layer
    elif isinstance(input_layer, models.Model):
        x = input_layer.inputs[0]
        input_layer = input_layer.output

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=5, strides=1,
                          padding='same', activation='relu',
                          name='conv1')(input_layer)

    # Reshape layer to be 1 capsule x [filters] atoms
    _, H, W, C = conv1.get_shape()
    conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=8,
                                    num_atoms=32, strides=1, padding='same',
                                    routings=1, name='primarycaps')(conv1_reshaped)

    # Layer 4: Convolutional Capsule: 1x1
    seg_caps = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=16,
                                strides=1, padding='same', routings=3,
                                name='seg_caps')(primary_caps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_seg = Length(num_classes=n_class, seg=True, name='out_seg')(seg_caps)

    # Decoder network.
    _, H, W, C, A = seg_caps.get_shape()

    if decoder:
        # Decoder network.
        prediction_shape = input_shape[:-1]+(1,) # assumed

        y = layers.Input(shape = prediction_shape, name='y')
        masked_by_y = Mask()([seg_caps, y])  # The true label is used to mask the output of capsule layer. For training
        masked = Mask()(seg_caps)  # Mask using the capsule with maximal length. For prediction

        def shared_decoder(mask_layer):
            reshaped_ = (H.value, W.value, C.value * A.value)
            recon_remove_dim = layers.Reshape(reshaped_)(mask_layer)

            recon_1 = layers.Conv2D(filters=64, kernel_size=1, padding='same',
                                    kernel_initializer='he_normal',
                                    activation='relu', name='recon_1')(recon_remove_dim)

            recon_2 = layers.Conv2D(filters=128, kernel_size=1, padding='same',
                                    kernel_initializer='he_normal',
                                    activation='relu', name='recon_2')(recon_1)

            out_recon = layers.Conv2D(filters=1, kernel_size=1, padding='same',
                                      kernel_initializer='he_normal',
                                      activation='sigmoid', name='out_recon')(recon_2)

            return out_recon

        # Models for training and evaluation (prediction)
        train_model = models.Model(inputs=[x, y],
                                   outputs=[out_seg, shared_decoder(masked_by_y)])
        eval_model = models.Model(inputs=x,
                                  outputs=[out_seg, shared_decoder(masked)])

    elif not decoder:
        # No decoder network (single input)
        train_model = models.Model(inputs=x, outputs=out_seg)
        eval_model = models.Model(inputs=x, outputs=out_seg)

    if add_noise:
        # manipulate model: Adding noise to the prediction and masks
        noise = layers.Input(shape=seg_caps.get_shape().as_list()[1:], name='noise')
        noised_seg_caps = layers.Add()([seg_caps, noise])
        masked_noised_y = Mask()([noised_seg_caps, y])
        if decoder: # (x,y)
            manipulate_model = models.Model(inputs=[x, y, noise],
                                            outputs=shared_decoder(masked_noised_y))
        elif not decoder: # (x,)
            manipulate_model = models.Model(inputs=[x, noise],
                                            outputs=shared_decoder(masked))
        return (train_model, eval_model, manipulate_model)
    else:
        return (train_model, eval_model)
