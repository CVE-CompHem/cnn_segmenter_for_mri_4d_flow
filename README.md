# cnn_segmenter_for_mri_4d_flow

To run training source the Python virtualenv in the random walker repository and run the following command e.g.

```
python train.py --train --config config/cnn_segmenter_cscs.json --model experiments/unet_cscs.json
```

to run inference using the same configuration simply replace `train` by `inference`:

```
python train.py --inference --config config/cnn_segmenter_cscs.json --model experiments/unet_cscs.json
```

