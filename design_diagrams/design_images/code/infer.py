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
    results_dir = config_data["results_dir"]
    models_path=config_data["models_path"]
    mapping_path=config_data["mapping_path"]
    num_point=config_data["num_point"]

    pointcloud_files = glob(os.path.join(project_dir, "*.las"))
    for file in pointcloud_files:
        list =[file]
        print(file)
        dataset = CustomDataset(list,is_training=False,label_map_file=mapping_path,num_point=num_point)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        num_classes = dataset.num_classes
        print(num_classes)

        model=PointNet2(num_classes) 
        model.load_state_dict(torch.load(models_path))
        model.to(device)

        predictions = infer_point_clouds(device,model, loader, num_classes)
        print(predictions.shape)
        resulting_point_cloud=np.vstack(predictions)
        print(resulting_point_cloud.shape)
        c=os.path.basename(file)
        save_to_las(resulting_point_cloud,os.path.join(results_dir,c))
        print(os.path.join(results_dir,c))

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Process configuration file")
        parser.add_argument('config_path', type=str, help='Path to the YAML configuration file')
        args = parser.parse_args()
        main(args.config_path)