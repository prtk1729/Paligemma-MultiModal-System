
# Common libs
import argparse
import os


# Dataset
from Config_standard import Config_standard
from torch.utils.data import DataLoader

from datasets.standard_dataset import *
from utils.config import Config
from utils.trainer import ModelTrainer
from models.architectures import KPFCNN


def run(config):
    GPU_ID = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    previous_training_path = ''
    chkp_idx = None
    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)

    else:
        chosen_chkp = None
    ##############
    # Prepare Data
    ##############
    print()
    print('Data Preparation')
    print('****************')
    # Initialize datasets
    training_dataset = STDataset(config, set='training', use_potentials=True)
    val_dataset = STDataset(config, set='validation', use_potentials=True)
    # Initialize samplers
    training_sampler = STSampler(training_dataset)
    test_sampler = STSampler(val_dataset)
    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=STCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)
    test_loader = DataLoader(val_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=STCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)
    # Calibrate samplers
    training_sampler.calibration(training_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)
    print('\nModel Preparation')
    print('*****************')
    # Define network model
    t1 = time.time()
    net = KPFCNN(config, training_dataset.label_values, training_dataset.ignored_labels)
    debug = False
    if debug:
        print('\n*************************************\n')
        print(net)
        print('\n*************************************\n')
        for param in net.parameters():
            if param.requires_grad:
                print(param.shape)
        print('\n*************************************\n')
        print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))
        print('\n*************************************\n')
    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))
    print('\nStart training')
    print('**************')
    # Training
    trainer.train(net, training_loader, test_loader, config)

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