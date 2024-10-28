
import numpy as np
import os
import open3d as o3d
import pylas
import os, numpy as np, open3d as o3d
from typing import  Dict, Any
import laspy

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


def write_las(pcd, las_file, local_origin=[0, 0, 0]):
        
        # Extract points and colors from the PointCloud object
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

         # Create a new header
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = np.min(points, axis=0)
        header.scales = np.array([0.1, 0.1, 0.1])
        
        # Create a LasWriter and a point record, then write it
        with laspy.open(las_file, mode="w", header=header) as writer:
            point_record = laspy.ScaleAwarePointRecord.zeros(points.shape[0], header=header)
            point_record.x = points[:, 0]
            point_record.y = points[:, 1]
            point_record.z = points[:, 2]

            # If colors are available, add them to the LAS file
            if colors is not None and len(colors) == len(points):
                colors = (colors * 65535.0).astype(np.uint16)  # Scale colors back to [0, 65535]
                point_record.red = colors[:, 0]
                point_record.green = colors[:, 1]
                point_record.blue = colors[:, 2]
            
            # Translate points back to their original position using the local_origin
            point_record.x += local_origin[0]
            point_record.y += local_origin[1]
            point_record.z += local_origin[2]
            writer.write_points(point_record)


def write_las(points: np.ndarray, colors: np.ndarray, classification: np.ndarray, las_file: str, local_origin=[0, 0, 0]):
    """
    Write points, their RGB colors, and classification to a LAS file.

    Parameters:
    - points: A numpy array of shape (N, 3) containing the XYZ coordinates of the points.
    - colors: A numpy array of shape (N, 3) containing the RGB colors of the points, values in [0, 1].
    - classification: A numpy array of shape (N,) containing an integer classification for each point.
    - las_file: Path to the output LAS file.
    - local_origin: The local origin coordinates to translate the points, default is [0, 0, 0].
    """
    # Create a new header with point_format 3 to include RGB information
    header = laspy.LasHeader(point_format=3, version="1.2")
   # header.offsets = np.min(points, axis=0)
    #header.scales = np.array([0.1, 0.1, 0.1])
    
    with laspy.open(las_file, mode="w", header=header) as writer:
        point_record = laspy.ScaleAwarePointRecord.zeros(len(points), header=header)
        point_record.x = points[:, 0] - local_origin[0]
        point_record.y = points[:, 1] - local_origin[1]
        point_record.z = points[:, 2] - local_origin[2]

        # Scale colors to the range [0, 65535] as expected by the LAS format
        if colors is not None and colors.size > 0:
            scaled_colors = (colors * 65535).astype(np.uint16)
            point_record.red = scaled_colors[:, 0]
            point_record.green = scaled_colors[:, 1]
            point_record.blue = scaled_colors[:, 2]

        # Assign classification to each point
        if classification is not None and classification.size > 0:
            point_record.classification = classification.astype(np.uint8)

        writer.write_points(point_record)
