
# Common libs
import os
from torch.utils.data import DataLoader
from utils.config import Config
from utils.trainer import ModelTrainer
import sys

# Dataset
from Config_standard import Config_standard
from datasets.standard_dataset import *
from models.architectures import KPFCNN


def run(config_path="kpconv_config.yml"):
    global config
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Charging on {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Marching on the CPU")
    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = device
 
    ##############
    # Prepare Data
    ##############
    print()
    print('Data Preparation')
    print('****************')
    # Initialize configuration class
    config = Config_standard(config_path)
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
    net = KPFCNN(config, training_dataset.label_values, training_dataset.ignored_labels)
    # Define a trainer class
    trainer = ModelTrainer(net, config)

    print('\nStart training')
    print('**************')
    # Training
    trainer.train(net, training_loader, test_loader, config)

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        config_path = sys.argv[1]
    else:
        config_path = "kpconv_config.yml"
    run(config_path)