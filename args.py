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
    parser.add_argument('--model', type=str, default='experiments/unet.json', help='Directory containing model configuration')

    # training arguments
    parser.add_argument('--training_input', type=str, help='Path to the directory containing the training data')
    parser.add_argument('--training_output', type=str, help='Path where the trainined CNN models are saved')

    # inference arguments
    parser.add_argument('--inference_input', type=str, help='Path to the FlowMRI image to be segmented')
    parser.add_argument('--inference_output', type=str, help='Path where the segmented SegmentedFlowMRI should be saved')

    # debug arguments
    parser.add_argument('--debug_server', type=str, help='Socket address (hostname:port) of Pycharm debug server')

    return parser.parse_args()

args = parse_args()

