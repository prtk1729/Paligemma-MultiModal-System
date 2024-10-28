from Model import PointNet2
from prod3d import*
from Dataset import CustomDataset
from glob import glob  
from torch.utils.data import DataLoader, Dataset
import argparse
import yaml

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Charging on {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Marching on the CPU")


def main(config_path):
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)

    project_dir = config_data["project_dir"]
    models_path=config_data["models_path"]
    mapping_path=config_data["mapping_path"]
    num_point=config_data["num_point"]
    train_dir=config_data["train_dir"]
    val_dir=config_data["val_dir"]
    epochs=config_data["epochs"]

    pointcloud_train_files = glob(os.path.join(train_dir, "*.las"))
    pointcloud_val_files = glob(os.path.join(val_dir, "*.las"))
    valid_list = pointcloud_val_files
    train_list = pointcloud_train_files

    print("############ DATASET ############")
    train_dataset = CustomDataset(train_list,label_map_file=mapping_path,num_point=num_point)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = CustomDataset(valid_list,label_map_file=mapping_path,is_training=False,num_point=num_point)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    num_classes = train_dataset.num_classes
    print(num_classes)
    model=PointNet2(num_classes) 
    model.to(device)
    print("############ Start training ############")

    train_and_save_model(device,
                        train_loader,
                        val_loader,
                          num_classes,model,  
                          save_path=models_path,
                            epochs =epochs )



if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Process configuration file")
        parser.add_argument('config_path', type=str, help='Path to the YAML configuration file')
        args = parser.parse_args()
        main(args.config_path)