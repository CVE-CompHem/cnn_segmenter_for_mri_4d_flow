# cnn_segmenter_for_mri_4d_flow

Steps for running the training on the Flownet images:
1. Pull Flownet images (fully-sampled as well as at different undersampling ratios) from Pollux.
2. Pull Random walker segmentations from Pollux (these serve as ground truths for training the CNN).
3. Run the "data_flownet_reorganize_dir_structure.sh" script. This will simplify the directory structure and file names.
   In this file, you will have to adapt the directory names according to your computer.
4. Run the "data_flownet_prepare_training_data.py" script. This will combine the training images and labels into 1 hdf5 file. 
   In this file, you will have to adapt the location where the file containing the training data needs to be stored.
5. Run the training command as given below.

To run training source the Python virtualenv in the random walker repository and run the following command e.g.

```
python train.py --training_input <path to hdf5 file containing the training data> --training_output <path where the training CNN model should be saved>
```
For an example, please see check the train.sh script.

To run inference, run the following command:

```
python inference.py --training_output <path where the training CNN model is saved> --inference_input <path of the FlowMRI test image> --inference_output <path where the SegmentedFlowMRI should be saved>
```
For an example, please see check the inference.sh script.

