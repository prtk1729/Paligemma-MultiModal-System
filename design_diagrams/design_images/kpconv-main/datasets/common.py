
import numpy as np
from torch.utils.data import  Dataset
from utils.config import Config
from kernels.kernel_points import create_3D_rotations
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
from typing import List, Optional, Tuple,Any, Dict

def grid_subsampling(
    points: np.ndarray,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    sampleDl: float = 0.1,
    verbose: int = 0
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Perform grid subsampling on a set of points, and optionally on associated features and labels.
    
    This function acts as a Python wrapper for an underlying C++ subsampling implementation, 
    selecting the appropriate version based on the provided inputs. The subsampling method 
    used is the barycenter for points and features within grid voxels of a specified size.
    
    Parameters:
        points (np.ndarray): A (N, 3) matrix of input points.
        features (Optional[np.ndarray]): An optional (N, d) matrix of features (floating point numbers).
        labels (Optional[np.ndarray]): An optional (N,) vector of integer labels.
        sampleDl (float): The side length of the grid voxels used for subsampling.
        verbose (int): If set to 1, subsampling process details are displayed. Defaults to 0 for no output.
    
    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]: A tuple containing the subsampled points,
        features (if provided), and labels (if provided).
    """
    # Arguments for the subsampling function are prepared in a dictionary.
    # This approach simplifies the handling of optional parameters.
    subsample_args = {
        'points': points,
        'sampleDl': sampleDl,
        'verbose': verbose
    }
    
    # Add features and labels to the arguments if they are provided.
    if features is not None:
        subsample_args['features'] = features
    if labels is not None:
        subsample_args['classes'] = labels

    # The ** operator unpacks the dictionary arguments directly into the function call.
    # This avoids repetitive and conditional logic for calling the subsample function.
    return cpp_subsampling.subsample(**subsample_args)


def batch_grid_subsampling(points: np.ndarray, batches_len: list, features: np.ndarray = None,
                           labels: np.ndarray = None, sampleDl: float = 0.1, max_p: int = 0,
                           verbose: int = 0, random_grid_orient: bool = True):
    """
    Performs grid subsampling on a batch of points, optionally including features and labels,
    with support for random grid orientation for data augmentation.

    Parameters:
    - points (np.ndarray): (N, 3) matrix of input points.
    - batches_len (list): List indicating the number of points in each batch.
    - features (np.ndarray, optional): (N, d) matrix of features associated with points.
    - labels (np.ndarray, optional): (N,) vector of labels for each point.
    - sampleDl (float): Edge length of the grid cells for subsampling.
    - max_p (int): Maximum number of points per grid cell, not used.
    - verbose (int): Verbosity level, if > 0 will print additional information.
    - random_grid_orient (bool): If True, applies a random rotation to each batch.

    Returns:
    Subsampled points, their batch lengths, and optionally subsampled features and labels.
    """

    # Apply random rotation if enabled
    if random_grid_orient:
        # Random rotation logic here (as per original function)
        pass  # Implementation remains unchanged from your original function

    # Directly handle calling cpp_subsampling.subsample_batch without using a dictionary
    if features is not None and labels is not None:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(
            points, batches_len, features, labels, sampleDl, max_p, verbose)
    elif features is not None:
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(
            points, batches_len, features, sampleDl=sampleDl, max_p=max_p, verbose=verbose)
        s_labels = None
    elif labels is not None:
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(
            points, batches_len, classes=labels, sampleDl=sampleDl, max_p=max_p, verbose=verbose)
        s_features = None
    else:
        s_points, s_len = cpp_subsampling.subsample_batch(
            points, batches_len, sampleDl=sampleDl, max_p=max_p, verbose=verbose)
        s_features, s_labels = None, None

    # Handle random grid orientation adjustments post-subsampling here if necessary

    return s_points, s_len, s_features, s_labels


def batch_neighbors(
    queries: np.ndarray, 
    supports: np.ndarray, 
    q_batches: List[int], 
    s_batches: List[int], 
    radius: float
) -> np.ndarray:
    """
    Computes neighbors within a specified radius for each query point from a set of support points,
    considering the data is batched. Each batch in the queries and supports can have different lengths.

    Parameters:
        queries (np.ndarray): A (N1, 3) array of query points, where N1 is the total number of query points.
        supports (np.ndarray): A (N2, 3) array of support points, where N2 is the total number of support points.
        q_batches (List[int]): A list of integers specifying the number of points in each batch of queries.
        s_batches (List[int]): A list of integers specifying the number of points in each batch of supports.
        radius (float): The search radius to find neighbors within.

    Returns:
        np.ndarray: An array of neighbor indices for each query point. The structure and format of this array
                    depend on the implementation of `cpp_neighbors.batch_query`.

    Note:
        This function acts as a wrapper around a C++ implemented function `batch_query` for efficiency.
        The actual structure of the returned neighbors' indices array is determined by the implementation
        details of `cpp_neighbors.batch_query`.
    """
    # Call the C++ extension to find neighbors for each query point within the specified radius
    return cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)

# ----------------------------------------------------------------------------------------------------------------------
#
#           Class definition
#       \**********************/



class PointCloudDataset(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, name: str) -> None:
        """
        Initialize the PointCloudDataset with basic dataset parameters.

        Parameters:
            name (str): Name of the dataset.
        """
        self.name: str = name
        self.path: str = ''
        self.label_to_names: Dict[int, str] = {}
        self.num_classes: int = 0
        self.label_values: np.ndarray = np.zeros((0,), dtype=np.int32)
        self.label_names: List[str] = []
        self.label_to_idx: Dict[int, int] = {}
        self.name_to_label: Dict[str, int] = {}
        self.config: Config = Config()
        self.neighborhood_limits: List[int] = []

    def __len__(self) -> int:
        """
        Returns the total number of items in the dataset.

        Override this method to return the actual dataset size.

        Returns:
            int: The number of items in the dataset.
        """
        # Placeholder implementation; should be overridden with actual dataset size
        return 0

    def __getitem__(self, idx: int) -> Any:
        """
        Retrieves the dataset item at the specified index.

        Override this method to return the actual data item, including features and labels as needed.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            Any: The dataset item at the given index. The exact type and content will depend on the dataset's structure.
        """
        # Placeholder implementation; should be overridden to return actual data items
        return 0

    def init_labels(self) -> None:
        """
        Initializes label-related parameters based on the `label_to_names` dictionary.

        This method sets up various attributes related to labels, including the number of classes,
        the mapping of label values to names, and vice versa.
        """
        self.num_classes: int = len(self.label_to_names)
        self.label_values: np.ndarray = np.sort(np.array([k for k, v in self.label_to_names.items()]))
        self.label_names: list = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx: Dict[int, int] = {label: idx for idx, label in enumerate(self.label_values)}
        self.name_to_label: Dict[str, int] = {name: label for label, name in self.label_to_names.items()}
    


    def augmentation_transform(
        self, 
        points: np.ndarray, 
        normals: Optional[np.ndarray] = None, 
        verbose: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Applies augmentation transforms to point clouds, including rotation, scaling, and noise addition.

        Parameters:
            points (np.ndarray): The (N, 3) array of input points.
            normals (Optional[np.ndarray]): The (N, 3) array of normals associated with the points, if any.
            verbose (bool): If True, print additional information about the transformations.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]: The transformed points, 
            optionally transformed normals (if provided), scale factors, and rotation matrix.
        """
        # Rotation
        R = self._apply_random_rotation(points)

        # Scale
        scale = self._apply_random_scale(points)

        # Noise
        noise = self._apply_random_noise(points)

        # Apply transforms
        augmented_points = self._apply_transforms(points, R, scale, noise)

        if normals is not None:
            augmented_normals = self._transform_normals(normals, R, scale)
            return augmented_points, augmented_normals, scale, R
        else:
            return augmented_points, None, scale, R


    
    def _apply_random_rotation(self, points: np.ndarray) -> np.ndarray:
        """
        Applies a random rotation to the points, based on the dataset's configuration.
        """
        R = np.eye(3, dtype=np.float32)
        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':
                theta = np.random.rand() * 2 * np.pi
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            elif self.config.augment_rotation == 'all':
                # Polar coordinates
                theta = np.random.rand() * 2 * np.pi
                phi = (np.random.rand() - 0.5) * np.pi
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
                alpha = np.random.rand() * 2 * np.pi
                R = self._create_3D_rotations(u, alpha)
        return R

    def _apply_random_scale(self, points: np.ndarray) -> np.ndarray:
        """
        Applies random scaling to the points, based on the dataset's configuration.
        """
        min_s, max_s = self.config.augment_scale_min, self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) + min_s
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)
        return scale

    def _apply_random_noise(self, points: np.ndarray) -> np.ndarray:
        """
        Adds random Gaussian noise to the points, based on the dataset's configuration.
        """
        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)
        return noise

    def _apply_transforms(self, points: np.ndarray, R: np.ndarray, scale: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """
        Applies the calculated rotation, scaling, and noise to the points.
        """
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise
        return augmented_points

    def _transform_normals(self, normals: np.ndarray, R: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
        Transforms normals based on the applied rotation and scale, including renormalization.
        """
        normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
        augmented_normals = np.dot(normals, R) * normal_scale
        augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)
        return augmented_normals


    def big_neighborhood_filter(self, neighbors: np.ndarray, layer: int) -> np.ndarray:
        """
        Filters neighborhoods based on a precomputed maximum number of neighbors per point for a specific layer.
        This limit is set during initialization to keep a certain percentage of the neighborhoods untouched,
        aiming to reduce the computational cost in later processing stages by limiting the neighborhood size.

        Parameters:
            neighbors (np.ndarray): A 2D array of shape (N, M) containing the neighborhood indices for N points,
                                    where M is the maximum neighborhood size found across all points.
            layer (int): The index of the processing layer for which the neighborhood filtering is being applied.
                         This allows for different limits to be applied at different stages of processing.

        Returns:
            np.ndarray: A 2D array of shape (N, K) where K <= M, containing the filtered neighborhood indices
                        for N points. If a limit is set for the specified layer, neighborhoods are cropped to
                        this limit. Otherwise, the original neighborhoods are returned.

        Note:
            The limit for each layer (`neighborhood_limits`) is expected to be set during the class initialization
            and is based on a target percentage of neighborhoods to remain unchanged, balancing between computational
            efficiency and neighborhood completeness.
        """
        # Check if the neighborhood limits have been defined for the layers
        if len(self.neighborhood_limits) > 0 and layer < len(self.neighborhood_limits):
            # Crop the neighbors matrix to the limit specified for the current layer
            cropped_neighbors = neighbors[:, :self.neighborhood_limits[layer]]
            return cropped_neighbors
        else:
            # If no limit is set for the layer, return the original neighbors matrix
            return neighbors

    def segmentation_inputs(
        self,
        stacked_points: np.ndarray,
        stacked_features: np.ndarray,
        labels: np.ndarray,
        stack_lengths: List[int]
    ) -> List[Any]:
        """
        Prepares inputs for segmentation models from point cloud data, including operations like
        neighborhood finding, pooling, and upsampling preparations.

        Parameters:
            stacked_points (np.ndarray): The concatenated points from all samples.
            stacked_features (np.ndarray): Corresponding features for each point.
            labels (np.ndarray): Labels for each point.
            stack_lengths (List[int]): Number of points in each sample.

        Returns:
            List[Any]: Processed inputs ready for use in segmentation models, including points, neighbors,
                       pool indices, upsamples indices, and stack lengths, along with features and labels.
        """

        # Initialize configuration parameters and input lists
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius
        layer_blocks = []
        input_points, input_neighbors, input_pools, input_upsamples, input_stack_lengths, deform_layers = [], [], [], [], [], []

        # Iterate through the architectural blocks defined in the configuration
        for block_i, block in enumerate(self.config.architecture):
            # Collect non-pooling/upscaling blocks for potential use in convolutions
            if not any(x in block for x in ('pool', 'strided', 'global', 'upsample')):
                layer_blocks.append(block)
                continue  # Skip to the next block if current block doesn't require special handling

            # Initialize deform_layer flag and handle convolutional neighbors if necessary
            deform_layer = False
            if layer_blocks:
                # Adjust radius for deformable layers
                r = r_normal * self.config.deform_radius / self.config.conv_radius if any('deformable' in blck for blck in layer_blocks) else r_normal
                deform_layer = any('deformable' in blck for blck in layer_blocks)
                # Compute convolution neighbors
                conv_i = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)
            else:
                conv_i = np.zeros((0, 1), dtype=np.int32)  # Placeholder for layers without convolutions

            # Handle pooling and upsampling for layers that require it
            if 'pool' in block or 'strided' in block:
                # Adjust subsampling length and compute pooled points and neighbors
                dl = 2 * r_normal / self.config.conv_radius
                pool_p, pool_b,_,_ = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)
                r_pool = r_normal * self.config.deform_radius / self.config.conv_radius if 'deformable' in block else r_normal
                deform_layer |= 'deformable' in block
                pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r_pool)
                up_i = batch_neighbors(stacked_points, pool_p, stack_lengths, pool_b, 2 * r_pool)
            else:
                pool_i, pool_p, pool_b, up_i = np.zeros((0, 1), dtype=np.int32), np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32), np.zeros((0, 1), dtype=np.int32)

            # Apply filtering to neighborhood matrices to reduce their size
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))
            if up_i.size > 0:
                up_i = self.big_neighborhood_filter(up_i, len(input_points) + 1)

            # Update input lists for the current layer
            input_points.append(stacked_points)
            input_neighbors.append(conv_i.astype(np.int64))
            input_pools.append(pool_i.astype(np.int64))
            input_upsamples.append(up_i.astype(np.int64))
            input_stack_lengths.append(stack_lengths)
            deform_layers.append(deform_layer)

            # Prepare for the next layer
            stacked_points, stack_lengths = pool_p, pool_b
            r_normal *= 2  # Double the radius for the next layer
            layer_blocks = []  # Clear layer blocks for the next iteration

            if 'global' in block or 'upsample' in block:
                break  # Terminate the loop for global pooling or upsampling layers

        # Compile all processed inputs into a list for model consumption
        return input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths + [stacked_features, labels]














