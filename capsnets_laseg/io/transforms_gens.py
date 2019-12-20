import numpy as np
import nibabel as nib
import os

from .preprocess_utils import get_positive_idx, get_random_slice_idx, \
                              reshape
from .base_gens import BaseTransformGenerator

class Transformed2DGenerator(BaseTransformGenerator):
    """
    Loads data, slices them based on the number of positive slice indices and
    applies data augmentation with `batchgenerators.transforms`.
    * Supports channels_last
    * .nii files should not have the batch_size dimension
    Attributes:
        list_IDs: list of filenames
        data_dirs: list of paths to both the input dir and labels dir
        batch_size: The number of images you want in a single batch
        n_pos: The number of positive class 2D images to include in a batch
        transform (Transform instance): If you want to use multiple Transforms,
            use the Compose Transform.
        max_patient_shape (tuple): representing the maximum patient shape in
            a dataset; i.e. ((z,)x,y)
            * Note: If you have 3D medical images and want 2D slices and don't
            want to overpad the slice dimension (z),
            provide a shape that is only 2D (x,y).
        step_per_epoch:
        pos_mask: boolean representing whether or not output the positive masks (X*Y)
            * If True, inputs are for capsule networks with a decoder.
            * If False, inputs are for everything else.
        shuffle: boolean
    """
    def __init__(self, list_IDs, data_dirs, batch_size=2, n_pos=1,
                 transform=None, max_patient_shape=(256, 320),
                 steps_per_epoch=1536, pos_mask=True, shuffle=True):

        BaseTransformGenerator.__init__(self, list_IDs=list_IDs,
                                        data_dirs=data_dirs,
                                        batch_size=batch_size,
                                        n_channels=1, n_classes=1, ndim=2,
                                        transform=transform,
                                        max_patient_shape=max_patient_shape,
                                        steps_per_epoch=steps_per_epoch,
                                        shuffle=shuffle)
        self.n_pos = n_pos
        self.pos_mask = pos_mask
        if n_pos == 0:
            print("WARNING! Your data is going to be randomly sliced.")
            self.mode = "rand"
        elif n_pos == batch_size:
            print("WARNING! Your entire batch is going to be",
                  "positively sampled.")
            self.mode = "pos"
        else:
            self.mode = "bal"

        if len(self.max_patient_shape) == 2:
            self.dynamic_padding_z = True # no need to pad the slice dimension

    def __getitem__(self, idx):
        """
        Defines the fetching and on-the-fly preprocessing of data.
        Args:
            idx: the id assigned to each worker
        Returns:
        if self.pos_mask is True:
            (X,Y): a batch of transformed data/labels based on the n_pos
            attribute.
        elif self.pos_mask is False:
            ([X, Y], [Y, pos_mask]): multi-inputs for the capsule network
            decoder
        """
        # file names
        max_n_idx = (idx + 1) * self.batch_size
        if max_n_idx > self.indexes.size:
            print("Adjusting for idx: ", idx)
            self.adjust_indexes(max_n_idx)

        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # Fetches batched IDs for a thread
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # balanced sampling
        if self.mode == "bal":
            # generating data for both positive and randomly sampled data
            X_pos, Y_pos = self.data_gen(list_IDs_temp[:self.n_pos],
                                         pos_sample=True)
            X_rand, Y_rand = self.data_gen(list_IDs_temp[self.n_pos:],
                                           pos_sample=False)
            # concatenating all the corresponding data
            X, Y = np.concatenate([X_pos, X_rand], axis=0), np.concatenate([Y_pos, Y_rand], axis=0)
            # shuffling the order of the positive/random patches
            out_rand_indices = np.arange(0, X.shape[0])
            np.random.shuffle(out_rand_indices)
            X, Y = X[out_rand_indices], Y[out_rand_indices]
        # random sampling
        elif self.mode == "rand":
            X, Y = self.data_gen(list_IDs_temp, pos_sample=False)
        elif self.mode == "pos":
            X, Y = self.data_gen(list_IDs_temp, pos_sample=True)
        # data augmentation
        if self.transform is not None:
            X, Y = self.apply_transform(X, Y)
        # print("Getting item of size: ", indexes.size, "out of ", self.indexes.size, "with idx: ", idx, "\nX shape: ", X.shape)
        assert X.shape[0] == self.batch_size, \
            "The outputted batch doesn't match the batch size."
        if self.pos_mask:
            pos_mask = X * Y
            return ([X, Y], [Y, pos_mask])
        elif not self.pos_mask:
            return (X, Y)

    def data_gen(self, list_IDs_temp, pos_sample):
        """
        Generates a batch of data.
        Args:
            list_IDs_temp: batched list IDs; usually done by __getitem__
            pos_sample: boolean on if you want to sample a positive image or not
        Returns:
            tuple of two numpy arrays: x, y
        """
        images_x = []
        images_y = []
        for id in list_IDs_temp:
            # loads data as a numpy arr and then changes the type to float32
            x_train = np.expand_dims(np.load(os.path.join(self.data_dirs[0], id)), -1)
            y_train = np.expand_dims(np.load(os.path.join(self.data_dirs[1], id)), -1)
            # Padding to the max patient shape (so the arrays can be stacked)
            if self.dynamic_padding_z: # for when you don't want to pad the slice dimension (bc that usually changes in images)
                pad_shape = (x_train.shape[0], ) + self.max_patient_shape
            elif not self.dynamic_padding_z:
                pad_shape = self.max_patient_shape
            x_train = reshape(x_train, x_train.min(), pad_shape + (self.n_channels,))
            y_train = reshape(y_train, 0, pad_shape + (self.n_classes, ))
            # extracting slice:
            if pos_sample:
                slice_idx = get_positive_idx(y_train)[0]
            elif not pos_sample:
                slice_idx = get_random_slice_idx(x_train)

            images_x.append(x_train[slice_idx]), images_y.append(y_train[slice_idx])

        input_data, seg_masks = np.stack(images_x), np.stack(images_y)
        return (input_data, seg_masks)
