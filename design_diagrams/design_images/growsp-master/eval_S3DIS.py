import torch
import torch.nn.functional as F
from datasets.S3DIS import S3DIStest, cfl_collate_fn_test
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
#from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
from scipy.optimize import linear_sum_assignment as linear_assignment
import open3d as o3d
from lib.helper_ply import read_ply as read_ply
import colorsys
from train_S3DIS import *
from sklearn.cluster import KMeans
from models.fpn import Res16FPN18
from lib.utils import get_fixclassifier
import warnings
import argparse
import os
warnings.filterwarnings('ignore')

###
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='data/S3DIS/input',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default='data/S3DIS/initial_superpoints',
                        help='initial superpoint path')
    parser.add_argument('--save_path', type=str, default='trained_models/S3DIS/',
                        help='model savepath')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--workers', type=int, default=10, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=6, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=300, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=12, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=128, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    return parser.parse_args()

def indices_to_colors(indices, nb_class):
    # Generate colors in HSV space and convert to RGB
    hsv_colors = [(i / nb_class, 1, 1) for i in range(nb_class)]
    rgb_colors = [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_colors]

    # Map class indices to colors
    color_map = {i: rgb_colors[i] for i in range(nb_class)}
    
    return np.array([color_map.get(idx, [0, 0, 0]) for idx in indices])

def eval_once_coords(args, model, test_loader, classifier, use_sp=False,results_dir = "./results"):
    os.makedirs(results_dir, exist_ok=True)

    all_preds, all_label = [], []

    for data in test_loader:
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region = data

            in_field = ME.TensorField(features, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            region = region.squeeze()
            #
            if use_sp:
                region_inds = torch.unique(region)
                region_feats = []
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        region_feats.append(feats[valid_mask].mean(0, keepdim=True))
                region_feats = torch.cat(region_feats, dim=0)
                #
                scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
                preds = torch.argmax(scores, dim=1).cpu()

                region_scores = F.linear(F.normalize(region_feats), F.normalize(classifier.weight))
                region_no = 0
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        preds[valid_mask] = torch.argmax(region_scores, dim=1).cpu()[region_no]
                        region_no +=1
            else:
                scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
                preds = torch.argmax(scores, dim=1).cpu()

            preds = preds[inverse_map.long()]

              # Read the original point cloud data
            idx=index[0]
            file=test_loader.dataset.file[idx]
            file_name_without_extension = os.path.splitext(os.path.basename(file))[0]

            original_data = read_ply(file)
            original_coords = np.vstack((original_data['x'], original_data['y'], original_data['z'])).T
            original_coords = original_coords.astype(np.float32)
            np_pred=(preds).numpy()
            np_labels=(labels).numpy()
            
            # Convert to Open3D point cloud for saving
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(original_coords)

            # Assuming predictions and labels are class indices
            # Convert them to colors for visualization (you might need a custom function)
            nb_class=len(test_loader.dataset.label_to_names)
            pred_colors = indices_to_colors((preds).numpy(),nb_class)
            label_colors = indices_to_colors((labels).numpy(),nb_class)

            # Set colors to the point cloud
            pcd.colors = o3d.utility.Vector3dVector(pred_colors)
            pcd_path = os.path.join(results_dir, f"pred_{file_name_without_extension}_{idx}.pcd")
            o3d.io.write_point_cloud(pcd_path, pcd)

            # Optionally, save labels in a similar way
            pcd.colors = o3d.utility.Vector3dVector(label_colors)
            label_pcd_path = os.path.join(results_dir, f"label_{file_name_without_extension}_{idx}.pcd")
            o3d.io.write_point_cloud(label_pcd_path, pcd)
            if original_coords.shape[0] == np_pred.shape[0] == np_labels.shape[0]:
                # Reshape np_pred and np_labels if they are not in the correct shape (1D to 2D with one column)
                if np_pred.ndim == 1:
                    np_pred = np_pred.reshape(-1, 1)
                if np_labels.ndim == 1:
                    np_labels = np_labels.reshape(-1, 1)

                # Concatenate the arrays
                point_cloud = np.concatenate((original_coords, np_pred, np_labels), axis=1)

                # Save to a text file
                np.savetxt(os.path.join(results_dir,f"results_{file_name_without_extension}_{idx}.txt", point_cloud, fmt='%f', delimiter=' '))

            else:
                print("Error: Arrays do not have the same length.")
            print(file_name_without_extension)

            all_preds.append(preds[labels!=args.ignore_label])
            all_label.append(labels[[labels!=args.ignore_label]])
    return all_preds, all_label
    #return all_preds, all_label



def eval_once(args, model, test_loader, classifier, use_sp=False):

    all_preds, all_label = [], []
    for data in test_loader:
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region = data

            in_field = ME.TensorField(features, coords, device=0)
            feats = model(in_field)
            feats = F.normalize(feats, dim=1)

            region = region.squeeze()
            #
            if use_sp:
                region_inds = torch.unique(region)
                region_feats = []
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        region_feats.append(feats[valid_mask].mean(0, keepdim=True))
                region_feats = torch.cat(region_feats, dim=0)
                #
                scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
                preds = torch.argmax(scores, dim=1).cpu()

                region_scores = F.linear(F.normalize(region_feats), F.normalize(classifier.weight))
                region_no = 0
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        preds[valid_mask] = torch.argmax(region_scores, dim=1).cpu()[region_no]
                        region_no +=1
            else:
                scores = F.linear(F.normalize(feats), F.normalize(classifier.weight))
                preds = torch.argmax(scores, dim=1).cpu()

            preds = preds[inverse_map.long()]
            all_preds.append(preds[labels!=args.ignore_label]), all_label.append(labels[[labels!=args.ignore_label]])

    return all_preds, all_label



def eval(epoch, args, test_areas = ['Area_5'],second_round=False):

    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
    if second_round:
            model.load_state_dict(torch.load(os.path.join(args.save_path, 'model2_' + str(epoch) + '_checkpoint.pth')))
    else:
        model.load_state_dict(torch.load(os.path.join(args.save_path, 'model_' + str(epoch) + '_checkpoint.pth')))
    model.eval()

    cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
    if second_round:
        cls.load_state_dict(torch.load(os.path.join(args.save_path, 'cls2_' + str(epoch) + '_checkpoint.pth')))
    else:
        cls.load_state_dict(torch.load(os.path.join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth')))
    cls.eval()

    primitive_centers = cls.weight.data###[300, 128]
    print('Merging Primitives')
    #cluster_pred = KMeans(n_clusters=args.semantic_class, n_init=10, random_state=0, n_jobs=10).fit_predict(primitive_centers.cpu().numpy())#.astype(np.float64))
    cluster_pred = KMeans(n_clusters=args.semantic_class, n_init=10, random_state=0).fit_predict(primitive_centers.cpu().numpy())

    '''Compute Class Centers'''
    centroids = torch.zeros((args.semantic_class, args.feats_dim))
    for cluster_idx in range(args.semantic_class):
        indices = cluster_pred ==cluster_idx
        cluster_avg = primitive_centers[indices].mean(0, keepdims=True)
        centroids[cluster_idx] = cluster_avg
    # #
    centroids = F.normalize(centroids, dim=1)
    classifier = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.semantic_class, centroids=centroids).cuda()
    classifier.eval()

    test_dataset = S3DIStest(args, areas=test_areas)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=cfl_collate_fn_test(), num_workers=4, pin_memory=True)

    preds, labels = eval_once(args, model, test_loader, classifier)
    all_preds = torch.cat(preds).numpy()
    all_labels = torch.cat(labels).numpy()

    '''Unsupervised, Match pred to gt'''
    sem_num = args.semantic_class
    mask = (all_labels >= 0) & (all_labels < sem_num)
    histogram = np.bincount(sem_num * all_labels[mask] + all_preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    '''Hungarian Matching'''
    # m = linear_assignment(histogram.max() - histogram)
    # o_Acc = histogram[m[:, 0], m[:, 1]].sum() / histogram.sum()*100.
    # m_Acc = np.mean(histogram[m[:, 0], m[:, 1]] / histogram.sum(1))*100
    # hist_new = np.zeros((sem_num, sem_num))
    # for idx in range(sem_num):
    #     hist_new[:, idx] = histogram[:, m[idx, 1]]

    # '''Final Metrics'''
    # tp = np.diag(hist_new)
    # fp = np.sum(hist_new, 0) - tp
    # fn = np.sum(hist_new, 1) - tp
    # IoUs = tp / (tp + fp + fn + 1e-8)
    # m_IoU = np.nanmean(IoUs)
    # s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    # for IoU in IoUs:
    #     s += '{:5.2f} '.format(100 * IoU)

    # return o_Acc, m_Acc, s
    # Hungarian Matching
    m = linear_assignment(histogram.max() - histogram)

    # Calculate overall accuracy
# Assuming m is a 2D array with shape (N, 2)
    row_indices, col_indices = m
    o_Acc = sum(histogram[row_indices[i], col_indices[i]] for i in range(len(row_indices))) / histogram.sum() * 100

    # Calculate mean accuracy
    row_indices, col_indices = m
    m_Acc = np.mean([histogram[row_indices[i], col_indices[i]] / histogram.sum(1)[row_indices[i]] for i in range(len(row_indices))]) * 100

    #m_Acc = np.mean([histogram[i, j] / histogram.sum(1)[i] for i, j in m]) * 100

    # Initialize new histogram for matched classes
    #hist_new = np.zeros((sem_num, sem_num))

    # Reassign the histogram based on the Hungarian matching
    #for i, j in m:
    #    hist_new[:, j] = histogram[:, i]

    hist_new = np.zeros((sem_num, sem_num))
    for i in range(len(row_indices)):
        hist_new[:, col_indices[i]] = histogram[:, row_indices[i]]


    # Calculate Final Metrics
    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)

    # Format the final metrics string
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)

    return o_Acc, m_Acc, s


# if __name__ == '__main__':

    # args = parse_args()
    # epoch = find_max_epoch_in_saved_checkpoints(args.save_path)
    # o_Acc, m_Acc, s = eval(epoch, args)
    # print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc), s)
if __name__ == '__main__':

    args = parse_args()
    for epoch in range(10, 1500):
        if epoch % 1270 == 0:
            o_Acc, m_Acc, s = eval(epoch, args)
            print('Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc), s)