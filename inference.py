from capsnets_laseg.inference.local_eval import pred_data_2D_per_sample, evaluate_2D
def goals():
    """
    * Create a huge function to run along with config parser
    [Maybe read in a config like NiftyNet?]
    Possible arguments:
    * model type
    * weights
    * x_dir, y_dir
    * csv with the fnames
    Optional arguments:
    * input_shape = (256, 320, 1)
    * ct = False
    * batch_size = 2 (should raise a warning to change it if the network is a CNN)
    ------------------------------------------------------------------------------
    Make a load_model function
    """
config = dict()
config['model_type'] = ""

def main():
    actual_y, pred = pred_data_2D_per_sample(eval_model, local_train_path, local_label_path, list_IDs['test'],\
                                            pad_shape = input_shape[:-1], batch_size = 2)
    results = evaluate_2D(actual_y, pred)

    pass
if __name__ == "__main__":
    main()
