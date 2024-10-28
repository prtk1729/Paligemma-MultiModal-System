# -*- coding: utf-8 -*-
"""
3D Point Cloud to 2D Image Generation

Created by Florent Poux, (c) 2023 Licence MIT
To reuse in your project, please cite the most appropriate article accessible on my Google Scholar page

Have fun with this script!
"""

#%% 1. import libraries

import torch
import open3d as o3d

#%% 2. test CUDA

torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name()

#%% 3. Load a point cloud and visualize

DATANAME = "../DATA/exterior_scan.ply"
pcd = o3d.io.read_point_cloud("../DATA/" + DATANAME)
o3d.visualization.draw_geometries([pcd])