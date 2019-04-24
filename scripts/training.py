import argparse
import os
import json

from training_utils import get_model, get_transforms, get_callbacks, get_generators

def add_bool_arg(parser, name, default=False):
    """
    From: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    Handles boolean cases from command line through the creating two mutually exclusive arguments: --name and --no-name.
    Args:
        parser (arg.parse.ArgumentParser): the parser you want to add the arguments to
        name: name of the common feature name for the two mutually exclusive arguments; dest = name
        default: default boolean for command line
    Returns:
        None
    """
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name:default})

if __name__ == "__main__":
    # parsing the arguments from the command prompt
    parser = argparse.ArgumentParser(description="For training on Task 2 of the Medical Segmentation Decathlon using Capsule Networks and CNNs")
    parser.add_argument("--weights_dir", type = str, required = True,
                        help = "Path to the base directory where you want to save your weights")
    parser.add_argument("--dset_path", type = str, required = True,
                        help = "Path to the base directory where the imagesTr and labelsTr directory are.")
    parser.add_argument("--model_name", type = str, required = True,
                        help = "Path to the base directory where you want to download and extract the tar file")
    add_bool_arg(parser, "decoder", default = False) # defaults to extract = True
    parser.add_argument("--epochs", type = int, required = True,
                        help = "Number of epochs")
    parser.add_argument("--batch_size", type = int, required = False, default = 2,
                        help = "Batch size for the CNN should be 17 and for the Capsule Network, it should be 2.")
    parser.add_argument("--n_pos", type = int, required = False, default = 1,
                        help = "Try to make this 1/3 of the batch size (exception for the Capsule Network because its batch size is too small.)")
    parser.add_argument("--lr", type = float, required = False, default = 3e-5,
                        help = "The learning rate")
    parser.add_argument("--steps_per_epoch", type = int, required = False, default = 1536,
                        help = "Number of batches per epoch.")
    parser.add_argument("--fold_json_path", type = str, required = False, default = "",
                        help = "Path to the json with the filenames split. If this is not specified, the json will be created in 'weights_dir.'")
    args = parser.parse_args()
    # Setting up the initial filenames and path
    data_dirs = [os.path.join(args.dset_path, "imagesTr"), os.path.join(args.dset_path, "labelsTr")]
    if args.fold_json_path == "":
        print("Creating the fold...60/20/20 split")
        from keras_med_io.utils.misc_utils import get_list_IDs
        id_dict = get_list_IDs(data_dirs[0])
        args.fold_json_path = os.path.join(args.weights_dir, args.model_name + "_fold1.json")
        print("Saving the fold in: ", args.fold_json_path)
        # Saving current fold as a json
        with open(args.fold_json_path, 'w') as fp:
            json.dump(id_dict, fp)
    else:
        with open(args.fold_json_path, 'r') as fp:
            id_dict = json.load(fp)

    # create generators, callbacks, and model
    transform = get_transforms()
    gen, gen_val = get_generators(id_dict, data_dirs, args.batch_size, args.n_pos, transform, steps = args.steps_per_epoch, pos_mask = args.decoder)
    model = get_model(args.model_name, args.lr, decoder = args.decoder)
    callbacks = get_callbacks(args.model_name, args.weights_dir, args.decoder)
    # training
    # feel free to change the settings here if you want to
    print("Starting training...")
    history = model.fit_generator(generator = gen, steps_per_epoch = len(gen), epochs = args.epochs, callbacks = callbacks, validation_data = gen_val,
                                        validation_steps = len(gen_val), max_queue_size = 20, workers = 1, use_multiprocessing = False)
    print("Finished training!")
    # save model and history
    history_path = os.path.join(args.weights_dir, args.model_name + "_history.json")
    with open(history_path, 'w') as fp:
        json.dump(history, fp)
    print("Saved the training history in ", history_path)

    weights_path = os.path.join(args.weights_dir, args.model_name + "_weights_" + str(args.epochs) + "epochs.h5")
    model.save(weights_path)
    print("Saved the weights in ", weights_path)
