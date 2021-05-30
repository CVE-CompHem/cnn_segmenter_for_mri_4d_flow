# cnn_segmenter_for_mri_4d_flow

Steps for running the training on the Flownet images:
1. Pull Flownet images (fully-sampled as well as at different undersampling ratios) from Pollux.
2. Pull Random walker segmentations from Pollux (these serve as ground truths for training the CNN).
3. Run the "data_flownet_reorganize_dir_structure.sh" script. This will simplify the directory structure and file names.
   This script takes three arguments, time_stamp_host of Flownet images without undersampling, time_stamp_host of Flownet images with different undersampling ratios and time_stamp_host of random walker segmentations of Flownet images. 
```
./data_flownet_reorganize_dir_structure.sh 2021-02-11_19-41-32_daint102 2021-03-19_15-46-05_daint102 2021-02-11_20-14-44_daint102
```

4. Run the "data_flownet_prepare_training_data.py" script. This will combine the training images and labels into one hdf5 file called training_data.hdf5. 
   This script takes one argument which indicates the storage location of this file.
``` 
python data_flownet_prepare_training_data.py ../../data/v1/decrypt/segmenter/segmenter_data/
```

5. Run the training command as given below.

```
python train.py --training_input <path to hdf5 file containing the training data> #../../data/v1/decrypt/segmenter/segmented_data/training_data.hdf5 --training_output <path where the training CNN model should be saved> # ../../data/v1/decrpyt/segmenter/cnn_segmenter/hpc_predict/v1/training/
```

To run inference, run the following command:

```
python inference.py --training_output <path where the training CNN model is saved> --inference_input <path of the FlowMRI data> --inference_output <path where the SegmentedFlowMRI data should be saved>
```


