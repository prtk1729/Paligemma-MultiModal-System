from torch.utils.data import DataLoader

import test_models_standard
import train_Standard
from Config_standard import Config_standard
from datasets.standard_dataset import *
from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPCNN, KPFCNN
from test_models_standard import *
from train_Standard import *
import signal
import os

# Dataset
from Config_standard import Config_standard
from torch.utils.data import DataLoader

from datasets.standard_dataset import *
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Charging on {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Marching on the CPU")

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        config_path = sys.argv[1]
    else:
        config_path = "kpconv_config.yml"

    config = Config_standard(config_path)
    chosen_log = config.saving_path
    model_path = get_model_path(chosen_log)
    print(model_path)
    if not os.path.exists(model_path):
        train_Standard.run(config_path)
    test_models_standard.run(config_path)
