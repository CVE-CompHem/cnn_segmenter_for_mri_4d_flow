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

    # training arguments
    parser.add_argument('--training-input', type=str,
                    help='Training input directory (for training only)')
    parser.add_argument('--training-output', type=str,
                    help='Training output directory (for training only)')

    # inference arguments
    parser.add_argument('--inference-input', type=str,
                    help='Input FlowMRI (for inference only)')
    parser.add_argument('--inference-output', type=str,
                    help='Output SegmentedFlowMRI (for inference only)')

    # debug arguments
    parser.add_argument('--debug_server', type=str,
                    help='Socket address (hostname:port) of Pycharm debug server')

    return parser.parse_args()

args = parse_args()

