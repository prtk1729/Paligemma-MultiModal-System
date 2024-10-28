
import numpy as np
import os
import open3d as o3d
import pylas
import csv
import yaml
import time
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from itertools import product
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import os, random, numpy as np, open3d as o3d
import pickle
from typing import  Dict, Any
from torch.utils.data import  Dataset
from scipy.spatial import KDTree
from itertools import product

def find_classification_file(base_path):
    if os.path.isfile(base_path):
        base_path, ext = os.path.splitext(base_path)
        last_name = os.path.basename(base_path)

    """Find a corresponding classification file in the same directory."""
    for file in os.listdir(os.path.dirname(base_path)):
        if last_name in file and ('label' in file or 'classification'in file  or 'classif' in file):
            return os.path.join(os.path.dirname(base_path), file)
    return None


def get_classification_data (np_points,file_path):
    classification_path=find_classification_file(file_path)
    if classification_path and os.path.isfile(classification_path):
        try:
            classifications = np.loadtxt(classification_path)
            if len(classifications) == np_points:
                return classifications
            else:
                print("Number of classifications does not match the number of points. Using default value -1.")
        except Exception as e:
            print(f"Error reading classification file {classification_path}: {e}")
    return None
def read_las_file(file_path):
    """
    Reads a LAS file and returns the data as a structured NumPy array,
    including combined RGB colors and classification if they exist.
    """
    try:
        # Load the LAS file
        las = pylas.read(file_path)

        # Determine the number of points in the LAS file
        num_points = len(las.points)

        # Define the data structure for the structured array
        data_types = [('coordinates', '3f4')]
        
        # Check if RGB data is present
        if 'red' in las.point_format.dimension_names and \
           'green' in las.point_format.dimension_names and \
           'blue' in las.point_format.dimension_names:
            data_types.append(('colors', '3f4'))

        # Check if classification data is present
        if 'classification' in las.point_format.dimension_names and not ( las.classification== 0).all():
            data_types.append(('classification', 'u1'))

        # Create an empty structured array with the specified data types
        structured_array = np.empty(num_points, dtype=data_types)

        # Populate coordinates
        structured_array['coordinates'] = np.column_stack([las.x, las.y, las.z])

        # Populate combined RGB colors if present
        if 'colors' in structured_array.dtype.names:
             structured_array['colors'] = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0  # Normalizing to [0, 1]

        # Populate classification if present
        if 'classification' in structured_array.dtype.names:
            structured_array['classification'] = las.classification

        return structured_array

    except Exception as e:
        print(f"Error reading LAS file {file_path}: {e}")
        return None

def read_ply_file(file_path):
    try:
        ply = o3d.io.read_point_cloud(file_path)
        points = np.asarray(ply.points)
        colors = np.asarray(ply.colors) if ply.has_colors() else None
        data_types = [('coordinates', '3f4')]
        if colors is not None:
            data_types.append(('colors', '3f4'))

        classification= get_classification_data(len(points), file_path)
        if classification is not None:
            data_types.append(('classification', 'i4'))

        structured_array = np.empty(len(points), dtype=data_types)
        structured_array['coordinates'] = points
        if colors is not None:
            structured_array['colors'] = colors

    
        if classification is not None:
            structured_array['classification'] = classification    
        return structured_array
    except Exception as e:
        print(f"Error reading PLY file {file_path}: {e}")
        return None
    
def read_txt_file(file_path):
    """
    Reads a TXT file and returns the data as a structured NumPy array.
    Assumes the file may contain coordinates, optional RGB colors, and optional classification.
    """
    try:
        # Read data from the TXT file
        data = np.loadtxt(file_path)

        # Ensure data is 2D (handles the single-line case)
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)

        # Check for RGB and classification
        has_rgb = data.shape[1] >= 6
        has_classification = data.shape[1] in [7, 10]

        # Define the data structure
        data_types = [('coordinates', '3f4')]
        if has_rgb:
            data_types.append(('colors', '3f4'))
        if has_classification:
            data_types.append(('classification', 'i4'))

        # Create the structured array
        structured_array = np.zeros(data.shape[0], dtype=data_types)
        structured_array['coordinates'] = data[:, :3]
        if has_rgb:
            structured_array['colors'] = data[:, 3:6]/255
        if has_classification:
            structured_array['classification'] = data[:, -1]

        return structured_array

    except Exception as e:
        print(f"Error reading TXT file {file_path}: {e}")
        return None
def read_pcd_file(file_path):
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        data_types = [('coordinates', '3f4')]
        if colors is not None:
            data_types.append(('colors', '3f4'))

        classification= get_classification_data(len(points), file_path)
        if classification is not None:
            data_types.append(('classification', 'i4'))

        structured_array = np.empty(len(points), dtype=data_types)
        structured_array['coordinates'] = points
        if colors is not None:
            structured_array['colors'] = colors

        if classification is not None:
            structured_array['classification'] = classification    
        return structured_array
    except Exception as e:
        print(f"Error reading PCD file {file_path}: {e}")
        return None

def read_file_auto(file_path):
    """
    Automatically reads a file based on its extension if it is not a "label" file.

    :param file_path: Path to the file.
    :return: Structured NumPy array with the data or None if the file is a "label" file.
    """
    # Check if the file path contains the word "label" (you can customize this check)
    if "label" in os.path.basename(file_path):
        print(f"Skipping '{file_path}' because it is a 'label' file.")
        return None

    # Get the file extension
    file_extension = os.path.splitext(file_path)[1].lower()

    # Read the file based on its extension
    if file_extension == ".las":
        return read_las_file(file_path)
    elif file_extension == ".txt":
        return read_txt_file(file_path)
    elif file_extension == ".pcd":
        return read_pcd_file(file_path)
    elif file_extension == ".ply":
        return read_ply_file(file_path)
    else:
        print(f"Unsupported file format: '{file_extension}'")
        return None


def denoise(point_cloud_data,nb_neighbors):
    """
    Denoise a point cloud using Open3D's statistical outlier removal.

    Args:
    point_cloud_data (numpy.ndarray): A structured numpy array with 'coordinates' field for point cloud data.

    Returns:
    numpy.ndarray: The denoised point cloud as a structured numpy array.
    """

    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data['coordinates'])

    # Apply statistical outlier removal
    # nb_neighbors defines how many neighbors are taken into account in order to calculate the mean distance for a given point
    # std_ratio is the standard deviation ratio, points with a distance larger than this will be considered as outliers
    nb_neighbors = 20
    std_ratio = 2.0
    denoised_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)

    # Select corresponding colors and classification
    if 'colors' in point_cloud_data.dtype.names:
        selected_colors = point_cloud_data['colors'][ind]
    if 'classification' in point_cloud_data.dtype.names:
        selected_classification = point_cloud_data['classification'][ind]

    # Create a new structured array for the denoised data
    dtype = [('coordinates', 'f4', 3)]
    if 'colors' in point_cloud_data.dtype.names:
        dtype.append(('colors', 'f4', 3))
    if 'classification' in point_cloud_data.dtype.names:
        dtype.append(('classification', 'u1'))

    denoised_data = np.zeros(len(denoised_pcd.points), dtype=dtype)
    denoised_data['coordinates'] = np.asarray(denoised_pcd.points)

    if 'colors' in point_cloud_data.dtype.names:
        denoised_data['colors'] = selected_colors
    if 'classification' in point_cloud_data.dtype.names:
        denoised_data['classification'] = selected_classification

    return denoised_data

def normalize_coordinates(point_cloud_data):
    # Normalize XY coordinates by subtracting the mean and Z coordinate by subtracting the minimum
    xy_mean = np.mean(point_cloud_data['coordinates'][:, :2], axis=0)
    z_min = np.min(point_cloud_data['coordinates'][:, 2])
    point_cloud_data['coordinates'][:, :2] -= xy_mean
    point_cloud_data['coordinates'][:, 2] -= z_min

    return point_cloud_data,xy_mean,z_min

def preprocess(source_folder_path, destination_folder_path, force=False):
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)

    for dirpath, dirnames, filenames in os.walk(source_folder_path):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            file_name, file_ext = os.path.splitext(file)

            # Check if the file is a point cloud and doesn't contain specific keywords
            if file_ext.lower() in ['.ply', '.pcd', '.las', '.txt'] and \
               all(keyword not in file_name.lower() for keyword in ['label', 'classif', 'classification']):
                relative_path = os.path.relpath(dirpath, source_folder_path)
                destination_dir = os.path.join(destination_folder_path, relative_path)
                if not os.path.exists(destination_dir):
                    os.makedirs(destination_dir)

                npy_file_path = os.path.join(destination_dir, file_name + '.npy')
                npy_matrix_path = os.path.join(destination_dir, file_name + '.csv')

                # Process the file if force is True or if the .npy file doesn't already exist
                if force or not os.path.exists(npy_file_path):
                    try:
                        point_cloud_data = read_file_auto(file_path)
                        point_cloud_data,xy_mean,z_min = normalize_coordinates(point_cloud_data)
                        point_cloud_data = denoise(point_cloud_data)

                        np.save(npy_file_path, point_cloud_data)
                        with open(npy_matrix_path, 'w', newline='') as file:
                            writer = csv.writer(file)
                            line = xy_mean.copy()  # Create a copy of xy_mean to avoid modifying the original list
                            line.append(z_min)    # Append z_min to the copy of xy_mean
                            writer.writerow(line)
                        print(f"preprocess {file_path} to {npy_file_path}")
                    except Exception as e:
                        print(f"Error preprocessing {file_path}: {e}")

def train_and_save_model(device,train_loader,val_loader, num_classes,model, save_path="model.pth",  epochs = 100):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs, labels,index = batch
            p =inputs
            p, labels = p.float(), labels.long()
            p, labels = p.to(device), labels.to(device)
            seg_pred = model(p)
            seg_pred = seg_pred.contiguous().view(-1, num_classes)
            labels = labels.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        total_correct = 0
        total_points = 0  
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels,index = batch
                p =inputs
                p, labels = p.float(), labels.long()
                p, labels = p.to(device), labels.to(device)
                seg_pred = model(p)
                seg_pred = seg_pred.contiguous().view(-1, num_classes)
                labels = labels.view(-1, 1)[:, 0]
                _, predicted = seg_pred.max(1)
                total_correct += (predicted == labels).sum().item()
                total_points += labels.size(0)
        accuracy = 100 * total_correct / total_points
        print(f"Epoch [{epoch+1}/{epochs}], Loss {loss:.4f}, Accuracy: {accuracy:.2f}%")
    torch.save(model.state_dict(), save_path)

def infer_point_clouds(device,model, test_loader, num_classes):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            # 1 INFERENCE
            inputs, _,index = batch
            p=inputs.data.numpy()
            p =inputs
            p = p.float()
            p = p.to(device)
            seg_pred = model(p)
            seg_pred = seg_pred.contiguous().view(-1, num_classes)
            _, predicted = seg_pred.max(1)
            # 2 BUILDING POINTCLOUD RESULTS
            for i in range(p.shape[0]): 
                cloud_points =test_loader.dataset.get_data(index[i])
                start_idx = i * cloud_points.shape[1]
                end_idx = start_idx + cloud_points.shape[1]
                cloud_predictions = predicted[start_idx:end_idx].cpu().numpy().reshape(-1, 1)
                point_cloud = cloud_points.numpy()
                point_cloud =point_cloud[:3].T
                combined = np.hstack((point_cloud, cloud_predictions))
                predictions.append(combined)

    return np.array(predictions)

def process_data_in_tiles(data, num_points, labels=None):
    # Construct KDTree using only x and y coordinates
    # Transposing is now removed to save memory
    xyz_data = data[:3, :].T  # Transpose if necessary to get the shape (n_points, 3)
    tree = KDTree(xyz_data)

    total_points = data.shape[1]
    total_area = (data[0, :].max() - data[0, :].min()) * (data[1, :].max() - data[1, :].min())
    tile_area = total_area * num_points / total_points
    tile_size = np.sqrt(tile_area)

    # Extract bounding box for the data
    x_min, y_min = data[:2, :].min(axis=1)
    x_max, y_max = data[:2, :].max(axis=1)

    x_coords = np.arange(x_min, x_max, tile_size)
    y_coords = np.arange(y_min, y_max, tile_size)
    default_z = 0
    for x, y in product(x_coords, y_coords):
        # Query KDTree for points within the tile
        tile_points_indices = tree.query_ball_point([x, y, default_z], tile_size)

        if tile_points_indices:
            n_indices = len(tile_points_indices)

            # Adjust indices count according to the required number of points
            if n_indices != num_points:
                replace = n_indices < num_points
                tile_points_indices = np.random.choice(tile_points_indices, num_points, replace=replace)

            tile_points = data[:, tile_points_indices]
            
            # Use generator to yield one tile at a time instead of storing all in memory
            tile_label = labels[tile_points_indices] if labels is not None else np.zeros(num_points)
            yield tile_points, tile_label

def create_uniform_grid_tiles(data, labels, points_per_tile):

    x_min, y_min = np.min(data[0, :]), np.min(data[1, :])
    x_max, y_max = np.max(data[0, :]), np.max(data[1, :])

    tiles = []
    tile_labels = []

    # Iterate over 1 meter by 1 meter grid
    for x in np.arange(x_min, x_max, 1):
        for y in np.arange(y_min, y_max, 1):
            # Select points within the current 1x1 meter tile
            in_tile = (data[0, :] >= x) & (data[0, :] < x + 1) & (data[1, :] >= y) & (data[1, :] < y + 1)

            tile_points = data[:, in_tile]
            tile_label = labels[in_tile]

            # Adjust the number of points in the tile
            num_points_in_tile = tile_points.shape[1]
            if num_points_in_tile > 0:
                if num_points_in_tile < points_per_tile:
                    # If fewer points, duplicate randomly
                    indices = np.random.choice(num_points_in_tile, points_per_tile, replace=True)
                elif num_points_in_tile > points_per_tile:
                    # If more points, sample without replacement
                    indices = np.random.choice(num_points_in_tile, points_per_tile, replace=False)
                else:
                    # If equal, use as is
                    indices = np.arange(num_points_in_tile)

                tile_points = tile_points[:, indices]
                tile_label = tile_label[indices]

                tiles.append(tile_points)
                tile_labels.append(tile_label)
        

    return tiles, tile_labels

def save_to_las(np_array, filename):
    # Create a new LAS object
    las = pylas.create()

    # Assuming the first three columns of np_array are X, Y, Z and the fourth column is classification
    las.x = np_array[:, 0]
    las.y = np_array[:, 1]
    las.z = np_array[:, 2]

    # Assuming classification is stored as integers
    las.classification = np_array[:, 3].astype(np.uint8)

    # Write the LAS file
    las.write(filename)
