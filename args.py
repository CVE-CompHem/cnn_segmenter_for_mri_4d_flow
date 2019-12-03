import argparse

# Parse data input and output directories
def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run CNN Segmenter for 4D flow MRIs.')
    parser.add_argument('--train', dest='train', action='store_true',
                    help='run training')
    parser.add_argument('--inference', dest='train', action='store_false',
                    help='run inference')
    parser.add_argument('--config', type=str, default='config/cnn_segmenter_neerav.json', # default='system/cnn_segmenter_neerav.json',
                    help='Directory containing MRI data set')
    parser.add_argument('--model', type=str, default='experiments/unet_neerav.json',
                    help='Directory containing model configuration')
    return parser.parse_args()

args = parse_args()

