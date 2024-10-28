import torch
import torch.nn.functional as F
from datasets.S3DIS import S3DIStest, cfl_collate_fn_test
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
#from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
from scipy.optimize import linear_sum_assignment as linear_assignment

from sklearn.cluster import KMeans
from models.fpn import Res16FPN18
from lib.utils import get_fixclassifier
import warnings
import argparse
import os
warnings.filterwarnings('ignore')
from train_S3DIS import parse_args,find_max_epoch_in_saved_checkpoints
from eval_S3DIS import eval_once_coords
import open3d as o3d
import numpy as np

if __name__ == '__main__':
    args = parse_args()
   
    test_areas = ['Area_5']

    with torch.no_grad():
        max_completed_epoch = find_max_epoch_in_saved_checkpoints(args.save_path)
        epoch=max_completed_epoch
        path_model=os.path.join(args.save_path, 'model_' + str(epoch) + '_checkpoint.pth')
        path_classif=os.path.join(args.save_path, 'cls_' + str(epoch) + '_checkpoint.pth')

        print(f"########### Use the model: \n {path_model}")
        print(f"########### Use the classifier: \n {path_model}")

        model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args).cuda()
        model.load_state_dict(torch.load(path_model))
        model.eval()

        cls = torch.nn.Linear(args.feats_dim, args.primitive_num, bias=False).cuda()
        cls.load_state_dict(torch.load(path_classif))
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

        preds, labels = eval_once_coords(args, model, test_loader, classifier,results_dir = "./results")
  