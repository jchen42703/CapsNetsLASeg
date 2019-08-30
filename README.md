# Capsule Networks for the Automated Segmentation of Left Atrium in Cardiac MRI
![alt text](https://github.com/jchen42703/CapsNetsLASeg/blob/master/images/la_003.gif "la_003.nii")

## Introduction
Comparing 2D capsule networks to 2D convolutional neural networks for automated left atrium segmentation using __Keras__ and __Tensorflow__. The proposed __U-CapsNet__ (adding a U-Net feature extractor for the SegCaps) is based off [Capsules for Object Segmentation](https://arxiv.org/pdf/1804.04241.pdf)'s SegCaps architecture. The convolutional neural networks used are basic U-Nets with no residual connections and are based off of the [nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation](https://arxiv.org/pdf/1809.10486.pdf), which actually won the challenge that this dataset was taken from.

### Dataset
The networks were trained and tested on the left atrium segmentation dataset (Task 2) from the [2018 Medical Segmentation Decathlon](http://medicaldecathlon.com/). It's a small dataset with only 20 labeled volumes and 10 test volumes collected and labeled by King College London. The volumes are comprised of mono-modal MRIs, which have corresponding binary groundtruths (left atrium and background). The voxel spacing was a constant `1.3700000047683716 mm by 1.25 mm by 1.25 mm` where the slice thickness was transposed to the top. The main experiments do not resample them because the spacing is constant throughout the dataset and anisotropic spacing does not really affect 2D neural nets.

## Installation/Setup
### Dependencies
These are automatically installed through the `Regular repository installation` below:
* numpy>=1.10.2
* keras
* tensorflow
* nibabel
* [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators)
* sklearn

Needs to be installed separately:
* [keras_med_io](https://github.com/jchen42703/keras_med_io/)

### Regular repository installation
```
git clone https://github.com/jchen42703/CapsNetsLASeg.git
cd CapsNetsLASeg
pip install .
```
### keras_med_io installation
(If you don't already have it installed)
```
git clone https://github.com/jchen42703/keras_med_io.git
cd keras_med_io
pip install .
```
### Dataset Installation
Run the following line in terminal to automatically download the dataset, `Task02_Heart`, in `dset_path` and create a new preprocessed dataset, `Preprocessed_Heart` in `output_path`.
```
python ./CapsNetsLASeg/scripts/download_preprocess.py --dset_path=dset_path --output_path=output_path
```
Please make sure to check the actual [script](https://github.com/jchen42703/CapsNetsLASeg/blob/master/scripts/download_preprocess.py) for more indepth documentation.

##  Quick Tutorial
An example of how the scripts can be run is in [`examples/Running_CapsNetsLASeg_Scripts_[Demo].ipynb`](https://github.com/jchen42703/CapsNetsLASeg/blob/master/examples/Running_CapsNetsLASeg_Scripts_%5BDemo%5D.ipynb). Also, the documentation for the scripts is mainly in each script directly in `scripts/`.

### Training
Run the script below with the appropriate arguments to train your desired model:  
```
python ./CapsNetsLASeg/scripts/training.py --weights_dir=weights_dir --dset_path=./Preprocessed_Heart --model_name=name --epochs=n_epochs
```
__Required Arguments:__
* `--weights_dir`: Path to the base directory where you want to save your weights (does not include the .h5 filename)
* `--dset_path`: Path to the base directory where the imagesTr and labelsTr directory are.
* `--model_name`: Name of the model you want to train
  * Either: `cnn`, `capsr3`, `ucapsr3`, or `cnn-simple`
* `--epochs`: Number of epochs

You can view the other optional arguments, such as `batch_size`, `n_pos`, `lr`, etc. in the original [script](https://github.com/jchen42703/CapsNetsLASeg/blob/master/scripts/training.py).

### Inference
Once you're done training, you can now predict and evaluate on your separated test set. Note that in `weights_dir`, you'll see `model_name_fold1.json`. This is a dictionary representing the file splits for a single fold of cross validation, and the script below will use that to predict and evaluate on the separated test set (different from the test set for the actual challenge, `imagesTs`).
```
!python ./CapsNetsLASeg/scripts/inference.py --weights_path=./weights.h5 --raw_dset_path=./Task02_Heart --model_name=name --fold_json_path="./capsnetslaseg_fold1.json" --batch_size=17 --save_dir="./pred"
```

__Required Arguments:__
* `--weights_path`: Path to the saved weights (a .h5 file).
* `--raw_dset_path`: Path to the base directory (`Task02_Heart`) where the unpreprocessed imagesTr and labelsTr directory are.
* `--model_name`: Name of the model you want to train
  * Either: `cnn`, `capsr3`, `ucapsr3`, or `cnn-simple`
* `--fold_json_path`: Path to the json with the filenames split.

Similar to the previous section, you can view the other optional arguments, such as `batch_size`, `save_dir`, `decoder`, etc. in the original [script](https://github.com/jchen42703/CapsNetsLASeg/blob/master/scripts/inference.py).

## Results
<table>
  <tbody>
    <tr>
      <!-- header row -->
      <th>Neural Network</th>
      <th align="center">Parameters</th>
      <th align="center">Test Dice</th>
      <th align="center">Weights</th>
    </tr>
    <!--row (person information)-->
    <tr>
      <td align="center">U-Net</td>
      <td align="center">27,671,926</td>
      <td align="center">0.89</td>
      <td align="center">https://drive.google.com/open?id=1G_0sgIig5wcJ-nrIpZdCsOB1uFXwaX23</td>
    </tr>
    <tr>
      <td align="center">U-Net (Baseline)</td>
      <td align="center">4,434,385</td>
      <td align="center">0.866</td>
      <td align="center">https://drive.google.com/open?id=1Xm-TV1apc_LK8wJrDZC5pBGXeirE5S57</td>
    </tr>
    <tr>
      <td align="center">U-CapsNet</td>
      <td align="center">4,542,400</td>
      <td align="center">0.876</td>
      <td align="center">https://drive.google.com/open?id=1ji0U9bd0GoLdvXwK9ARTwpUum1NiNiQ-</td>
    </tr>
    <tr>
      <td align="center">SegCaps</td>
      <td align="center">1,416,112</td>
      <td align="center">0.81</td>
      <td align="center">https://drive.google.com/open?id=1k8f474s4rNwggtp3SWRTQXfLY-f85zvh</td>
    </tr>
  </tbody>
</table>

## Expanding to Other Datasets
Note that this repository is specifically catered towards the binary segmentation of mono-modal MRIs. However, the `AdaptiveUNet` architecture and the loss functions in [metrics](https://github.com/jchen42703/CapsNetsLASeg/blob/master/capsnets_laseg/models/metrics.py) can be extended to multi-class problems.

## About Keras

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

* allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
* supports both convolutional networks and recurrent networks, as well as combinations of the two.
* supports arbitrary connectivity schemes (including multi-input and multi-output training).
* runs seamlessly on CPU and GPU.

_Read the documentation: [Keras.io](http://keras.io/)_

Keras is compatible with: Python 2.7-3.5.
