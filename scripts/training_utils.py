import numpy as np
from batchgenerators.transforms import MirrorTransform, SpatialTransform, \
                                       Compose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from capsnets_laseg.io import Transformed2DGenerator
from capsnets_laseg.models import U_CapsNet, CapsNetR3, AdaptiveUNet, \
                                  SimpleUNet
from capsnets_laseg.metrics import dice_coefficient_loss, dice_hard, \
                                   dice_plus_xent_loss

def get_transforms(patch_shape=(256, 320), other_transforms=None,
                   random_crop=False):
    """
    Initializes the transforms for training.
    Args:
        patch_shape:
        other_transforms: List of transforms that you would like to add
            (optional). Defaults to None.
        random_crop (boolean): whether or not you want to random crop or center
            crop. Currently, the Transformed3DGenerator
        only supports random cropping. Transformed2DGenerator supports
        both random_crop=True and False.
    """
    ndim = len(patch_shape)
    spatial_params = {
        "do_elastic_deform": True, "alpha": (0., 1500.), "sigma": (30., 80.),
        "do_rotation": True, "angle_z": (0, 2 * np.pi), "do_scale": True,
        "scale": (0.75, 2.), "border_mode_data": "nearest",
        "border_cval_data": 0, "order_data": 1, "random_crop": random_crop,
        "p_el_per_sample": 0.1, "p_scale_per_sample": 0.1,
        "p_rot_per_sample": 0.1)
    }
    spatial_transform = SpatialTransform(patch_shape, **spatial_params)
    mirror_transform = MirrorTransform(axes=(0,1))
    transforms_list = [spatial_transform, mirror_transform]
    if other_transforms is not None:
        transforms_list = transforms_list + other_transforms
    composed = Compose(transforms_list)
    return composed

def get_generators(list_IDs, data_dirs, batch_size=2, n_pos=1, transform=None,
                   max_patient_shape=(256, 320), steps=1536, pos_mask=True):
    """
    Returns the training and validation generators.
    Args:
        Please refer to the `Transformed2DGenerator` documentation.
    Returns:
        gen, gen_val: 2D slice generators based on keras.utils.Sequence
    """
    print("Using 2D Generators...",
          f"\nUsing at least: {n_pos} positive class slices")
    train_ids, val_ids = list_IDs["train"], list_IDs["val"]
    gen = Transformed2DGenerator(train_ids, data_dirs, batch_size=batch_size,
                                 n_pos=n_pos, transform=transform,
                                 max_patient_shape=max_patient_shape,
                                 steps_per_epoch=steps, pos_mask=pos_mask)
    gen_val = Transformed2DGenerator(val_ids, data_dirs, batch_size=batch_size,
                                     n_pos=n_pos, transform=transform,
                                     max_patient_shape=max_patient_shape,
                                     steps_per_epoch=int(steps//6),
                                     pos_mask=pos_mask, shuffle=False)
    print(f"Steps per epoch: {len(gen)}\nValidation Steps: {len(gen_val)}")
    return gen, gen_val

def get_model(model_name, lr=3e-5, input_shape=(256, 320, 1), decoder=False,
              inference=False):
    """
    Creates and compiles a specified model.
    Args:
        model_name (str): either "cnn", "ucapsr3", "capsr3", or "cnn-simple"
        lr (float): Learning rate. Defaults to 3e-5. Suggested learning rates:
            CNNs: lr = 5e-4, Capsule Networks: lr = 3e-5
    Returns:
        A compiled model of your choice
    """
    # setting the common parameters
    opt = Adam(lr=lr, beta_1=0.99, beta_2=0.999, decay=0.0)

    if decoder:
        recon_wei = 0.2
        loss = {"out_seg": dice_coefficient_loss, "out_recon": "mse"}
        loss_weighting = {"out_seg": 1., "out_recon": recon_wei}
        metrics = {"out_seg": dice_hard}
    else:
        loss = dice_coefficient_loss
        loss_weighting = None
        metrics = [dice_hard]
    # actually building the models
    if model_name == "ucapsr3":
        ucaps = U_CapsNet(input_shape, n_class=2, decoder=decoder)
        train_model, eval_model = ucaps.build_model(model_layer="simple",
                                                    capsnet_type="r3",
                                                    upsamp_type="subpix")
    elif model_name == "capsr3":
        train_model, eval_model = CapsNetR3(input_shape, n_class=2,
                                            decoder=decoder,
                                            upsamp_type="subpix")
    elif model_name == "cnn":
        adap = AdaptiveUNet(2, input_shape, n_classes=1, max_pools=6,
                            starting_filters=15)
        train_model = adap.build_model(include_top=True)
        eval_model = None
        loss = dice_plus_xent_loss
        lr = 5e-4
    elif model_name == "cnn-simple":
        train_model = SimpleUNet(include_top=True, input_shape=input_shape)
        eval_model = None
        loss = dice_plus_xent_loss
        lr = 5e-4
    else:
        raise Exception("Specified model is not supported.")
    # compiling only the training model
    train_model.compile(optimizer=opt, loss=loss, metrics=metrics,
                        loss_weights=loss_weighting)
    if inference:
        return train_model, eval_model
    else:
        return train_model

def get_callbacks(model_name, checkpoint_dir="/content/checkpoint.h5",
                  decoder=False):
    """
    Returns a list of callbacks.
    Args:
        checkpoint_dir (str, path): make sure to have the actually .h5 file in
            the path
        decoder (boolean): whether or not your model uses the decoder
    Returns:
        a list of callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
    """
    if model_name == "ucapsr3" or model_name == "capsr3":
        mode = "max"
    else:
        mode = "min"
    # callback parameters
    min_delta = 5e-3
    cooldown = 0
    min_lr = 0
    patience_lr = 25 # patience for ReduceLROnPlateau
    patience_stop = 30 # patience for EarlyStopping
    # specifying the mode and corresponding metric
    if decoder:
        if mode == "min":
            monitor = "val_loss"
        elif mode == "max":
            monitor = "val_out_seg_dice_hard"
    elif not decoder:
        if mode == "min":
            monitor = "val_loss"
        elif mode == "max":
            monitor = "val_dice_hard"
    # initializing callbacks
    ckpoint = ModelCheckpoint(checkpoint_dir, monitor=monitor,
                              save_best_only=True, period=3)
    stop = EarlyStopping(monitor=monitor, min_delta=min_delta,
                         patience=patience_stop, mode=mode,
                         restore_best_weights=True)
    lrplat = ReduceLROnPlateau(monitor=monitor, factor=0.8,
                               patience=patience_lr, mode=mode,
                               min_delta=min_delta, min_lr=min_lr,
                               cooldown=cooldown)
    callbacks = [ckpoint, stop, lrplat]
    return callbacks

def get_list_IDs(data_dir, splits = [0.6, 0.2, 0.2]):
    """
    Divides filenames into train/val/test sets
    Args:
        data_dir: file path to the directory of all the files; assumes labels and training images have same names
        splits: a list with 3 elements corresponding to the decimal train/val/test splits; [train, val, test]
    Returns:
        a dictionary of file ids for each set
    """
    id_list = os.listdir(data_dir)
    total = len(id_list)
    train = round(total * splits[0])
    val_split = round(total * splits[1]) + train
    return {"train": id_list[:train], "val": id_list[train:val_split], "test": id_list[val_split:]
           }

def add_bool_arg(parser, name, default=False):
    """
    From: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    Handles boolean cases from command line through the creating two mutually
    exclusive arguments: --name and --no-name.
    Args:
        parser (arg.parse.ArgumentParser): the parser you want to add the
            arguments to
        name: name of the common feature name for the two mutually exclusive
            arguments; dest = name
        default: default boolean for command line
    Returns:
        None
    """
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=name, action="store_true")
    group.add_argument("--no-" + name, dest=name, action="store_false")
    parser.set_defaults(**{name:default})
