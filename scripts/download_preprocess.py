import os
import tarfile
from google_drive_downloader import GoogleDriveDownloader as gdd
import argparse
from capsnets_laseg.io import LocalPreprocessingBinarySeg
import glob

def dload_heart(dset_path='/content/'):
    """
    Args:
        dset_path: where you want to save the dataset
    """
    dataset = "Task02_Heart"
    id = '1wEB2I6S6tQBVEPxir8cA5kFB8gTQadYY'

    download_dataset(dset_path, dataset, id)
    return

def download_dataset(dset_path, dataset,
                     id='1wEB2I6S6tQBVEPxir8cA5kFB8gTQadYY'):
    """
    Args:
        dset_path: where you want to save the dataset
            (doesn't include the name)
        dataset: Dataset name
    """
    tar_path = os.path.join(dset_path, dataset) + '.tar'
    gdd.download_file_from_google_drive(file_id=id,
                                        dest_path=tar_path, overwrite=False,
                                        unzip=False)

    if not os.path.exists(os.path.join(dset_path, dataset)):
        print('Extracting data [STARTED]')
        tar = tarfile.open(tar_path)
        tar.extractall(dset_path)
        print('Extracting data [DONE]')
    else:
        print('Data already downloaded. Files are not extracted again.')

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="For downloading and preprocessing the dataset.")
    parser.add_argument("--dset_path", type=str, required=True,
                        help="Path to the base directory where you want to save your dataset (Doesn't include the dataset name)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to the base directory where you want to save your preprocessed dataset (Doesn't include the dataset name)")
    args = parser.parse_args()
    input_dir = os.path.join(args.dset_path, "Task02_Heart")
    output_dir = os.path.join(args.output_path, "Preprocessed_Heart")
    print("Input Directory: ", input_dir, "\nOutput Directory: ", output_dir)
    # Downloading the dataset if it's not already downloaded
    if not os.path.isdir(input_dir):
        dload_heart(dset_path=args.dset_path)
    preprocess = LocalPreprocessingBinarySeg(input_dir, output_dir)
    # removing the weird files that start with .__
    training_dir, labels_dir = os.path.join(input_dir, 'imagesTr'), os.path.join(input_dir, 'labelsTr')
    for filename in glob.glob(training_dir + "/._*"):
        os.remove(filename)
    for filename in glob.glob(labels_dir + "/._*"):
        os.remove(filename)
    # Preprocessing the dataset
    preprocess.gen_data()
