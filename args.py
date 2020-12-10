import argparse

# ================================================
# Parse data input and output directories
# ================================================
def parse_args():

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run CNN Segmenter for 4D flow MRIs.')
    parser.add_argument('--train', dest='train', action='store_true', help='run training')
    parser.add_argument('--inference', dest='train', action='store_false', help='run inference')
    parser.add_argument('--config', type=str, default='config/cnn_segmenter_neerav.json', help='Directory containing MRI data set')
    parser.add_argument('--model', type=str, default='experiments/unet_neerav.json', help='Directory containing model configuration')

    # training arguments
    parser.add_argument('--training_input', type=str, help='Training input directory (for training only)')
    parser.add_argument('--training_output', type=str, help='Training output directory (for training as well as inference)')

    # inference arguments
    parser.add_argument('--inference_input', type=str, help='Input FlowMRI (for inference only)')
    parser.add_argument('--inference_output', type=str, help='Output SegmentedFlowMRI (for inference only)')

    # debug arguments
    parser.add_argument('--debug_server', type=str, help='Socket address (hostname:port) of Pycharm debug server')

    return parser.parse_args()

args = parse_args()

