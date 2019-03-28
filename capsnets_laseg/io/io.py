from keras_med_io.utils.shape_io import reshape, resample_array, extract_nonint_region
from capsnets_laseg.io.io_utils import *
import json
import os
import nibabel as nib
from os.path import join, isdir
import numpy as np

class LocalPreprocessingBinarySeg(object):
    """
    Preprocessing for only binary segmentation tasks from the MSDs
    """
    def __init__(self, task_path, output_path, mean_patient_shape = (115, 320, 232)):
        self.task_path = task_path
        self.output_path = output_path
        self.mean_patient_shape = mean_patient_shape
        if not isdir(output_path):
            os.mkdir(output_path)
            print("Created directory: ", output_path)
        print(self.scan_metadata())

    def scan_metadata(self):
        """
        Checking which modalities the input has (CT/No CT) and whether the dataset is
        compatible with this class (binary).
        """
        # Automatically checking if the task is a CT dataset or not
        json_path = join(self.task_path, "dataset.json")
        with open(json_path) as json_data:
            jsondata = json.load(json_data)
        if jsondata['modality']['0'].lower() == 'ct':
            self.ct = True
        else:
            self.ct = False
        assert len(jsondata['labels'].keys()) == 2 # binary only
        return "Metadata scanning completed!"

    def gen_data(self):
        """
        Generates and saves preprocessed data
        Args:
            task_path: file path to the task directory (must have the corresponding "dataset.json" in it)
        Returns:
            preprocessed input image and mask
        """
        # Generating the data
        images_path, labels_path = join(self.task_path, "imagesTr"), join(self.task_path, "labelsTr")
        data_ids = os.listdir(images_path) # labels and training data have the same name
        # Generating data and saving them recursively
        for id in data_ids:
            image, label = nib.load(join(images_path, id)), nib.load(join(labels_path, id))
            orig_spacing = image.header['pixdim'][1:4]
            preprocessed_img, preprocessed_label = isensee_preprocess(image, label, orig_spacing, get_coords = False, ct = self.ct, \
                                              mean_patient_shape = self.mean_patient_shape)
            out_fnames = self.save_imgs(preprocessed_img, preprocessed_label, id)
            print(out_fnames)

    def save_imgs(self, image, mask, id):
        """
        Saves an image and mask pair as .npy arrays in the MSD file structure
        Args:
            image: numpy array
            mask: numpy array
            id: filenames
        Returns:
            A string: 'Saved' + patient_id (without the file subfix)
        """
        # saving the generated dataset
        # output dir in MSD format
        out_images_dir, out_labels_dir = join(self.output_path, "imagesTr"), join(self.output_path, "labelsTr")
        # checking to make sure that the output directories exist
        if not isdir(out_images_dir):
            os.mkdir(out_images_dir)
            print("Created directory: ", out_images_dir)
        if not isdir(out_labels_dir):
            os.mkdir(out_labels_dir)
            print("Created directory: ", out_labels_dir)
        # Saving the arrays as 'raw_id.npy'
        raw_id = id.split(".")[0] # gets rid of file stem
        np.save(os.path.join(out_images_dir, raw_id), image), np.save(os.path.join(out_labels_dir, raw_id), mask)
        return "Saved: " + raw_id
