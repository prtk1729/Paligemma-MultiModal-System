
# Basic libs
import torch
import torch.nn as nn
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import json
from sklearn.neighbors import KDTree
import open3d as o3d
# reader
from loader import *
# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from sklearn.metrics import confusion_matrix
from tqdm import tqdm  # Correctly import the tqdm function
class ModelTester:
    def __init__(self, net, chkp_path=None, on_gpu=True):
        """
        Initializes the model with the option to load from a checkpoint and set the device.

        Parameters:
        - net: The neural network model.
        - chkp_path (str, optional): Path to the checkpoint file. Default is None.
        - on_gpu (bool, optional): Flag to indicate if training should be on GPU. Default is True.
        """

        # Choose to train on CPU or GPU based on availability and user preference
        self.device = torch.device("cuda:0" if on_gpu and torch.cuda.is_available() else "cpu")
        net.to(self.device)

        # Load previous checkpoint if path is provided
        if chkp_path:
            try:
                if torch.cuda.is_available():
                    checkpoint = torch.load(chkp_path)
                else:
                    # Load checkpoint on CPU if CUDA isn't available
                    checkpoint = torch.load(chkp_path, map_location=lambda storage, loc: storage)
                
                net.load_state_dict(checkpoint['model_state_dict'])
                self.epoch = checkpoint['epoch']
                net.eval()  # Set the network to evaluation mode
                
                print("Model and training state restored from", chkp_path)
            except FileNotFoundError:
                print(f"Checkpoint file at '{chkp_path}' not found. Initializing model without loading checkpoint.")
            except KeyError as e:
                print(f"Invalid checkpoint format: {e}. Could not load the model state.")
        else:
            print("No checkpoint path provided. Initializing model without loading checkpoint.")

    def _create_directories(self, base_path, subdirs):
        """
        Create subdirectories in the base path if they do not already exist.

        Parameters:
        - base_path: The base directory path.
        - subdirs: A list of subdirectory names to create.
        """
        for subdir in subdirs:
            path = join(base_path, subdir)
            if not exists(path):
                makedirs(path)

    def _calculate_validation_proportions(self, test_loader, nc_model):
        """
        Calculate the proportions of each class in the validation set.

        Parameters:
        - test_loader: DataLoader for the test dataset.
        - nc_model: Number of classes in the model.

        Returns:
        - An array of proportions for each class.
        """
        val_proportions = np.zeros(nc_model, dtype=np.float32)
        i = 0
        for label_value in test_loader.dataset.label_values:
            if label_value not in test_loader.dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_value)
                                             for labels in test_loader.dataset.validation_labels])
                i += 1
        return val_proportions
    def update_test_probs(self, s_points: np.ndarray, stacked_probs: np.ndarray, in_inds: np.ndarray, cloud_inds: np.ndarray, lengths: list, config) -> None:
        """
        Updates the test probabilities with predictions for each point in the dataset.

        Parameters:
        - s_points: Subsampled points used for the current predictions.
        - stacked_probs: Probabilities predicted by the model for the subsampled points.
        - in_inds: Indices of the subsampled points within the original cloud.
        - cloud_inds: Indices of the point clouds that each batch belongs to.
        - lengths: List of lengths of each batch in the subsampled points.
        - config: Configuration object containing parameters like test_radius_ratio and test_smooth.
        """
        i0 = 0
        for b_i, length in enumerate(lengths):
            # Extract data for current batch
            points = s_points[i0:i0 + length]
            probs = stacked_probs[i0:i0 + length]
            inds = in_inds[i0:i0 + length]
            c_i = cloud_inds[b_i]

            # Apply radius filter if necessary
            if 0 < config.test_radius_ratio < 1:
                mask = np.sum(points ** 2, axis=1) < (config.test_radius_ratio * config.in_radius) ** 2
                inds = inds[mask]
                probs = probs[mask]

            # Update current probabilities in whole cloud
            self.test_probs[c_i][inds] = config.test_smooth * self.test_probs[c_i][inds] + (1 - config.test_smooth) * probs
            i0 += length
    
    def save_predictions(self,test_loader, proj_probs, test_path):
        """
        Save the predictions, potentials, and point clouds to disk.

        Parameters:
        - test_loader: DataLoader for the test dataset.
        - proj_probs: Projected probabilities for each class and each point in the dataset.
        - test_path: Base path where the predictions, potentials, and clouds are saved.
        """

        for i, file_path in tqdm(enumerate(test_loader.dataset.files), total=len(test_loader.dataset.files), desc="Saving Files"):

            # Load evaluation points and colors
            points, colors = test_loader.dataset.load_evaluation_points(file_path)

            # Get the predicted labels
            preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

            # Prepare file and directory names
            cloud_name = file_path.replace("\\", "/")
            cloud_name = os.path.basename(cloud_name)  # Extracts the file name from file_path
            base_name, _ = os.path.splitext(cloud_name)  # Splits the file name into name and extension

            test_name = join(test_path, 'predictions', base_name + '.las')
            
            # Save LAS file for the predicted point cloud
            write_las(points, colors, preds, test_name)

            # Save ASCII predictions for test set
            if test_loader.dataset.set == 'test':
                ascii_name = join(test_path, 'predictions', base_name + '.txt')
                np.savetxt(ascii_name, preds, fmt='%d')

    def cloud_segmentation_test(self, net, test_loader, config):
        """
        Test method for cloud segmentation models
        """
        
        num_votes= config.num_votes
        softmax = torch.nn.Softmax(dim=1)  # Softmax layer for probability distribution

        # Determine the number of classes predicted by the model from the configuration
        nc_model = config.num_classes

        # Initialize a global prediction array for all test clouds
        self.test_probs = [np.zeros((l.shape[0], nc_model)) for l in test_loader.dataset.input_labels]

        # Set up directories for saving test results if saving is enabled in the config
        if config.saving:
            # Ensure the saving path is in a consistent format
            test_path = config.path_output

            # Create necessary directories for saving test outcomes
            self._create_directories(test_path, ['predictions', 'probs', 'potentials'])
        else:
            test_path = None  # No saving path if saving is not enabled

        # Calculate class proportions for validation if applicable
        if test_loader.dataset.set == 'validation':
            val_proportions = self._calculate_validation_proportions(test_loader, nc_model)
        else:
            val_proportions = None

        #####################
        # Network predictions
        #####################

        test_epoch = 0
        last_min = 0

        # Start test loop
        with tqdm(total=num_votes) as pbar:
            while last_min < num_votes:
                for i, batch in enumerate(test_loader):
                    # New time
                    if 'cuda' in self.device.type:
                        batch.to(self.device)

                    # Forward pass
                    outputs = net(batch, config)


                    # Get probs and labels
                    stacked_probs = softmax(outputs).cpu().detach().numpy()
                    s_points = batch.points[0].cpu().numpy()
                    lengths = batch.lengths[0].cpu().numpy()
                    in_inds = batch.input_inds.cpu().numpy()
                    cloud_inds = batch.cloud_inds.cpu().numpy()

                    # Get predictions and labels per instance
                    self.update_test_probs( s_points, stacked_probs, in_inds, cloud_inds, lengths, config)

                # Update minimum od potentials
                new_min = torch.min(test_loader.dataset.min_potentials)
                #print('Test {:d}, end. Min potential = {:.4f}'.format(test_epoch, new_min))

                # Save predicted cloud
                if last_min + 1 < new_min:
                    # Update last_min
                    last_min += 1
                    pbar.update(1)
                    # Update the postfix message with the current test epoch and new_min value
                    pbar.set_postfix_str('{:d}: Min potential = {:.4f}'.format(test_epoch, new_min))
        
                    if test_loader.dataset.set == 'validation':
                        print('\nConfusion on sub clouds')
                        Confs = []
                        for i, file_path in enumerate(test_loader.dataset.files):

                            # Insert false columns for ignored labels
                            probs = np.array(self.test_probs[i], copy=True)
                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    probs = np.insert(probs, l_ind, 0, axis=1)

                            # Predicted labels
                            preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

                            # Targets
                            targets = test_loader.dataset.input_labels[i]

                            # Confs
                            Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                        # Regroup confusions
                        C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                        # Remove ignored labels from confusions
                        for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                            if label_value in test_loader.dataset.ignored_labels:
                                C = np.delete(C, l_ind, axis=0)
                                C = np.delete(C, l_ind, axis=1)

                        # Rescale with the right number of point per class
                        C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                        # Compute IoUs
                        IoUs = IoU_from_confusions(C)
                        mIoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * mIoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        print(s + '\n')


                test_epoch += 1

        # Project predictions
        print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
        proj_probs = []
        for i, file_path in enumerate(test_loader.dataset.files):
            # Reproject probs on the evaluations points
            probs = self.test_probs[i][test_loader.dataset.test_proj[i], :]
            proj_probs += [probs]

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                if label_value in test_loader.dataset.ignored_labels:
                    proj_probs[i] = np.insert(proj_probs[i], l_ind, 0, axis=1)

        print("Saving results in :",test_path)
        self.save_predictions(test_loader, proj_probs, test_path)
        