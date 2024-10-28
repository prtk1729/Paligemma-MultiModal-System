import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import os, random, numpy as np, open3d as o3d
import pickle
from typing import  Dict, Any
from torch.utils.data import  Dataset
from scipy.spatial import KDTree
from itertools import product
from prod3d import *

class CustomDataset(Dataset):
    def cloud_loader(self,pcd_name):
        cloud_data =read_las_file(pcd_name)
        coords=cloud_data['coordinates']
        gt=cloud_data['classification']
        return coords.T,gt
   
    def tile_point_cloud(self, data, labels, num_points):
        return create_uniform_grid_tiles(data,labels,num_points)
    
    def load_data_and_labels(self, files, has_ground_trust, num_point):
        data_list = []
        labels_list = []

        for file in files:
            cloud_data, gt = self.cloud_loader(file)

            tiles, tile_labels = self.tile_point_cloud(cloud_data, gt if has_ground_trust else None, num_point)


            # Convert tiles and tile_labels to PyTorch tensors
            tiles = [torch.tensor(tile).float() for tile in tiles]
            tile_labels = [torch.tensor(label).long() for label in tile_labels]

            data_list.extend(tiles)
            labels_list.extend(tile_labels)
        return data_list, labels_list

    def preprocess(self, cloud_data):
        cloud_data = cloud_data.clone()  # Avoid modifying the original data
        min_f = torch.min(cloud_data, dim=1).values
        mean_f = torch.mean(cloud_data, dim=1)
        correction = torch.tensor([mean_f[0], mean_f[1], min_f[2]])[:, None]
        cloud_data[0:3] -= correction
        return cloud_data

    def __init__(self, files, has_ground_trust=True,is_training=True, num_point=4096, label_map_file="label_mapping02.pkl"):
        self.data, self.labels = self.load_data_and_labels(files,  has_ground_trust, num_point)
        self.inputs = [self.preprocess(cloud_data) for cloud_data in self.data]
        if has_ground_trust:
            if is_training:
                 # Concatenate all tensors of labels to get all labels in a single tensor
                all_labels = torch.cat(self.labels) 
                
                # Get unique labels
                unique_labels = np.unique(all_labels)
                
                # Create label mapping
                self.label_mapping = {label: i for i, label in enumerate(unique_labels)}
                
                # Save the mapping in a file
                with open(label_map_file, "wb") as f:
                    pickle.dump(self.label_mapping, f)
            else:
                  with open(label_map_file, "rb") as f:
                    self.label_mapping = pickle.load(f)
            self.num_classes = len(self.label_mapping.keys())
            for i in range(0,len(self.labels)):
                self.labels[i] =  torch.tensor([self.label_mapping[label.item()] for label in self.labels[i]])
        else:
            self.labels =[]
            for i in range(0,len(self.inputs)):
                self.labels.append(torch.empty_like(self.inputs[0]))
            self.num_classes = 0

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index], index

    def __len__(self):
        return len(self.inputs)

    def get_data(self, index):
        return self.data[index]
