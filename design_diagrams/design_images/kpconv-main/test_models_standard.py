# Common libs
import signal
import numpy as np
import argparse
import os
import torch

# Dataset

from torch.utils.data import DataLoader
from Config_standard import Config_standard
from datasets.standard_dataset import *
from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPCNN, KPFCNN

def run(config):
    GPU_ID = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    chosen_log = config.saving_path
    config.load(chosen_log)
    chosen_chkp = get_model_path(chosen_log)

    print()
    print('Data Preparation')
    print('****************')
    test_dataset = STDataset(config, set='test', use_potentials=True)
    test_sampler = STSampler(test_dataset)
    collate_fn = STCollate
    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)
    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)
    print('\nModel Preparation')
    print('*****************')
    # Define network model
    t1 = time.time()
    net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))
    print('\nStart test')
    print('**********\n')
    tester.cloud_segmentation_test(net, test_loader, config)


def get_model_path(chosen_log, chkp_idx=None):
    try:
        # Find all checkpoints in the chosen training folder
        chkp_path = os.path.join(chosen_log, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']
        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)
        return chosen_chkp
    except:
        print("No trained model found")
        return ""


def override_config(config, args):
    """Override configuration parameters with command-line arguments."""
    if args.num_votes is not None:
        config.num_votes = args.num_votes

    if args.dataset_path_train is not None:
        config.dataset_path_train = args.dataset_path_train

    if args.dataset_path_test is not None:
        config.dataset_path_test = args.dataset_path_test

    if args.dataset_path_validation is not None:
        config.dataset_path_validation = args.dataset_path_validation

    if args.path_output is not None:
        config.path_output = args.path_output

    if args.saving_path is not None:
        config.saving_path = args.saving_path

    return config
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to prepare and run model with configuration.")
    parser.add_argument('--config_path', nargs='?', default="kpconv_config.yml", help="Path to the configuration file.")
    parser.add_argument('--num_votes', type=int, help="Override num_votes.")
    parser.add_argument('--dataset_path_train', help="Override dataset path for training.")
    parser.add_argument('--dataset_path_test', help="Override dataset path for testing.")
    parser.add_argument('--dataset_path_validation', help="Override dataset path for validation.")
    parser.add_argument('--path_output', help="Override output path.")
    parser.add_argument('--saving_path', help="Override saving path.")
    args = parser.parse_args()
    config = Config_standard(args.config_path)
    config = override_config(config, args)
    run(config)