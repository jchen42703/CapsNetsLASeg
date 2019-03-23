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
            preprocessed_img, preprocessed_label = self.preprocess_fn(image, label, orig_spacing)
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

    def preprocess_fn(self, input_image, mask, orig_spacing, get_coords = False):
        """
        Order:
        1) Cropping to non-zero regions
        2) Resampling to the median voxel spacing of the respective dataset
        3) Normalization

        Args:
            input_image:
            mask:
            task_path: file path to the task directory (must have the corresponding "dataset.json" in it)
            mean_patient_shape: obtained from Table 1. in the nnU-Net paper
            get_coords: boolean on whether to return extraction coords or not
        Returns:
            preprocessed input image and mask
        """
        input_image, mask = self._nii_to_np(input_image), self._nii_to_np(mask)
        # 1. Cropping
        if get_coords:
            extracted_img, extracted_mask, coords = extract_nonint_region(input_image, mask = mask, outside_value = 0, coords = get_coords)
        elif not get_coords:
            extracted_img, extracted_mask = extract_nonint_region(input_image, mask = mask, outside_value = 0, coords = False)
        # 2. Resampling
        transposed_spacing = orig_spacing[::-1] # doing so because turning into numpy array moves the batch dimension to axis 0
        med_spacing = [np.median(transposed_spacing) for i in range(3)]
        resamp_img = resample_array(extracted_img, transposed_spacing, med_spacing, is_label = False)
        resamp_label = resample_array(extracted_mask, transposed_spacing, med_spacing, is_label = True)
        # 3. Normalization
        norm_img = zscore_isensee(resamp_img, ct = self.ct, mean_patient_shape = self.mean_patient_shape)
        if get_coords:
            return (norm_img, resamp_label, coords)
        elif not get_coords:
            return (norm_img, resamp_label)

    def _nii_to_np(self, nib_img):
        """
        Converts a 3D nifti image to a numpy array of (z, x, y) dims
        """
        return np.transpose(nib_img.get_fdata(), [-1, 0, 1])
