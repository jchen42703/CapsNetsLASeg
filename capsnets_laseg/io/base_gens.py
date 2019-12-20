import os
import numpy as np
import nibabel as nib
import keras

class BaseGenerator(keras.utils.Sequence):
    """
    Basic framework for generating thread-safe data in keras.
    (no preprocessing and channels_last)
    Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    Attributes:
      list_IDs: filenames (.nii files); must be same for training and labels
      data_dirs: list of [training_dir, labels_dir]
      batch_size: int of desired number images per epoch
      n_channels: <-
      n_classes: <-
      shuffle: boolean on whether or not to shuffle the dataset
    """
    def __init__(self, list_IDs, data_dirs, batch_size, n_channels, n_classes,
                 shuffle=True):
        # lists of paths to images
        self.list_IDs = list_IDs
        self.data_dirs = data_dirs
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / float(self.batch_size)))

    def __getitem__(self, idx):
        '''
        Defines the fetching and on-the-fly preprocessing of data.
        '''
        # file names
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, Y = self.data_gen(list_IDs_temp)
        return (X, Y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # self.img_idx = np.arange(len(self.x))
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_gen(self, list_IDs_temp):
        '''
        Preprocesses the data
        Args:
            list_IDs_temp: temporary batched list of ids (filenames)
        Returns
            x, y
        '''
        raise NotImplementedError

class BaseTransformGenerator(BaseGenerator):
    """
    Loads data and applies data augmentation with `batchgenerators.transforms`.
    Attributes:
        list_IDs: list of filenames
        data_dirs: list of paths to both the input dir and labels dir
        batch_size: The number of images you want in a single batch
        n_channels: number of channels
        n_classes: number of unique labels excluding the background class (i.e. binary; n_classes = 1)
        ndim: number of dimensions of the input (excluding the batch_size and n_channels axes)
        transform (Transform instance): If you want to use multiple Transforms, use the Compose Transform.
        max_patient_shape: a tuple representing the maximum patient shape in a dataset; i.e. (x,y, (z,))
        steps_per_epoch: steps per epoch during training (number of samples per epoch = steps_per_epoch * batch_size )
        shuffle: boolean on whether to shuffle the dataset between epochs
    """
    def __init__(self, list_IDs, data_dirs, batch_size, n_channels, n_classes, ndim,
                transform=None, max_patient_shape=None, steps_per_epoch=1000, shuffle=True):

        BaseGenerator.__init__(self, list_IDs=list_IDs, data_dirs=data_dirs, batch_size=batch_size,
                               n_channels=n_channels, n_classes=n_classes, shuffle=shuffle)
        self.ndim = ndim
        self.transform = transform
        self.max_patient_shape = max_patient_shape
        if max_patient_shape is None:
            self.max_patient_shape = self.compute_max_patient_shape()
        n_samples = len(self.list_IDs)
        self.indexes = np.arange(n_samples)
        n_idx = self.batch_size * steps_per_epoch # number of samples per epoch
        # Handles cases where the dataset is small and the batch size is high
        if n_idx > n_samples:
            print("Adjusting the indexes since the total number of required samples (steps_per_epoch * batch_size) is greater than",
            "the number of provided images.")
            self.adjust_indexes(n_idx)
            print("Done!")
        assert self.indexes.size == n_idx

    def adjust_indexes(self, n_idx):
        """
        Adjusts self.indexes to the length of n_idx.
        """
        assert n_idx > self.indexes.size, "WARNING! The n_idx should be larger than the current number of indexes or else \
                                           there's no point in using this function. It has been automatically adjusted for you."
        # expanding the indexes until it passes the threshold: max_n_idx (extra will be removed later)
        while n_idx > self.indexes.size:
            self.indexes = np.repeat(self.indexes, 2)
        remainder = (len(self.indexes) % (n_idx))
        if remainder != 0:
            self.indexes = self.indexes[:-remainder]

        try:
            assert n_idx == self.indexes.size, "Expected number of indices per epoch does not match self.indexes.size."
        except AssertionError:
            raise Exception("Please make your steps_per_epoch > 3 if your batch size is < 3.")

    def __len__(self):
        """
        Steps per epoch (total number of samples per epoch / batch size)
        """
        return int(np.ceil(len(self.indexes) / float(self.batch_size)))

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        # self.img_idx = np.arange(len(self.x))
        # self.indexes = np.arange(len(self.indexes))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        """
        Defines the fetching and on-the-fly preprocessing of data.
        Returns a batch of data (x,y)
        """
        # file names
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # Fetches batched IDs for a thread
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, Y = self.data_gen(list_IDs_temp)
        if self.transform is not None:
            X, Y = self.apply_transform(X, Y)
        return (X, Y)

    def apply_transform(self, X, Y):
        data_dict = {}
        # batchgenerator transforms only accept channels_first
        data_dict["data"] = self.convert_dformat(X, convert_to="channels_first")
        data_dict["seg"] = self.convert_dformat(Y, convert_to="channels_first")
        data_dict = self.transform(**data_dict)
        # Desired target models accept channels_last dformat data (change this for your needs as you'd like)
        return (self.convert_dformat(data_dict["data"], convert_to="channels_last"),
                self.convert_dformat(data_dict["seg"], convert_to="channels_last"))

    def data_gen(self, list_IDs_temp):
        """
        Generates a batch of data.
        Args:
            list_IDs_temp: batched list IDs; usually done by __getitem__
            pos_sample: boolean on if you want to sample a positive image or not
        Returns:
            tuple of two numpy arrays: x, y
        """
        raise NotImplementedError

    def convert_dformat(self, arr, convert_to="channels_last"):
        """
        Args:
            arr: numpy array of shape (batch_size, x, y(,z), n_channels) (could be 4D or 5D)
            convert_to: desired data format to convert `arr` to; either "channels_last" or "channels_first"
        Returns:
            the transposed numpy array
        """
        # converting to channels_first
        if convert_to == "channels_first":
            if self.ndim == 2:
                axes_list = [0, -1, 1,2]
            elif self.ndim == 3:
                axes_list = [0, -1, 1, 2, 3]
        # converting to channels_last
        elif convert_to == "channels_last":
            if self.ndim == 2:
                axes_list = [0, 2, 3, 1]
            elif self.ndim == 3:
                axes_list = [0, 2, 3, 4, 1]
        else:
            raise Exception("Please choose a compatible data format: 'channels_last' or 'channels_first'")
        return np.transpose(arr, axes_list)

    def compute_max_patient_shape(self):
        """
        Computes various shape statistics (min, max, and mean) and ONLY returns the max_patient_shape
        Args:
            ...
        Returns:
            max_patient_shape: tuple representing the maximum patient shape
        """
        print("Computing shape statistics...")
        # iterating through entire dataset
        shape_list = []
        for id in self.list_IDs:
            x_train = load_data(os.path.join(self.data_dirs[0], id))
            shape_list.append(np.asarray(x_train.shape))
        shapes = np.stack(shape_list)
        # computing stats
        max_patient_shape = tuple(np.max(shapes, axis=0))
        mean_patient_shape = tuple(np.mean(shapes, axis=0))
        min_patient_shape = tuple(np.min(shapes, axis=0))
        print("Max Patient Shape: ", max_patient_shape, "\nMean Patient Shape: ", mean_patient_shape,
        "\nMin Patient Shape: ", min_patient_shape)
        # Running a quick check on a possible fail case
        try:
            assert len(max_patient_shape) == self.ndim
        except AssertionError:
            print("Excluding the channels dimension (axis = -1) for the maximum patient shape.")
            max_patient_shape = max_patient_shape[:-1]
        return max_patient_shape

def load_data(data_path, file_format=None):
    """
    Args:
        data_path: path to the image file
        file_format: str representing the format as shown below:
            * 'npy': data is a .npy file
            * 'nii': data is a .nii.gz or .nii file
            * Defaults to None; if it is None, it auto checks for the format
    Returns:
        A loaded numpy array (into memory) with type np.float32
    """
    assert os.path.isfile(data_path), "Please make sure that `data_path` is to a file!"
    # checking for file formats
    if file_format is None:
        if '.nii.gz' in data_path[-7:] or '.nii' in data_path[-4:]:
            file_format = 'nii'
        elif '.npy' in data_path[-4:]:
            file_format = 'npy'
    # loading the data
    if file_format == 'npy':
        return np.load(data_path).astype(np.float32)
    elif file_format == 'nii':
        return nib.load(data_path).get_fdata().astype(np.float32)
    else:
        raise Exception("Please choose a compatible file format: `npy` or `nii`.")
