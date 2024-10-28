# -*- coding: utf-8 -*-
"""
3D Deep Learning: Semantic Segmentation
PointNet 2+3/4 - Model Training
(c) Florent Poux, Licence: MIT.
"""

#%% 1. Environment Set-up

#Base Math libraries
import numpy as np

#Drawing libraries
import matplotlib.pyplot as plt
import open3d as o3d

#Deep Learning Libraries
import torch
import torch.nn.functional as nnf
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

#Utilities libraries
import copy
from glob import glob 
import os
import functools
import mock
from tqdm.auto import tqdm
import time

# Check if CUDA is available and set PyTorch to use GPU or CPU accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#%% 2. Point Cloud Data Preparation
""" In this step, 
*  We load the previously prepared data
*  We define the same function that was used for data preparation, to load a point 
*   We check if we can access labels of one element from the train_set
"""

#Your should navigate and extract the path to your project directory
project_dir="../DATA/AERIAL_LOUHANS"
pointcloud_train_files = glob(os.path.join(project_dir, "train/*.txt"))
pointcloud_test_files = glob(os.path.join(project_dir, "test/*.txt"))

# We create a function that load and normalize a point cloud tile
def cloud_loader(tile_name, features_used):
  cloud_data = np.loadtxt(tile_name).transpose()

  min_f=np.min(cloud_data,axis=1)
  mean_f=np.mean(cloud_data,axis=1)

  features=[]
  if 'xyz' in features_used:
    n_coords = cloud_data[0:3]
    n_coords[0] -= mean_f[0]
    n_coords[1] -= mean_f[1]
    n_coords[2] -= min_f[2]
    features.append(n_coords)
  if 'rgb' in features_used:
    colors = cloud_data[3:6]
    features.append(colors)
  if 'i' in features_used:
    IQR = np.quantile(cloud_data[-2],0.75)-np.quantile(cloud_data[-2],0.25)
    n_intensity = ((cloud_data[-2] - np.median(cloud_data[-2])) / IQR)
    n_intensity -= np.min(n_intensity)
    features.append(n_intensity)
  
  gt = cloud_data[-1]
  gt = torch.from_numpy(gt).long()
  cloud_data = torch.from_numpy(np.vstack(features))
  return cloud_data, gt

#Prepare the data in a train set, a validation set (to tune the model parameters) and a test set (to evaluate the performances)
#The validation is made of a random 20% of the train set.
valid_index = np.random.choice(len(pointcloud_train_files),int(len(pointcloud_train_files)/5), replace=False)
valid_list = [pointcloud_train_files[i] for i in valid_index]
train_list = [pointcloud_train_files[i] for i in np.setdiff1d(list(range(len(pointcloud_train_files))),valid_index)]
test_list = pointcloud_test_files
print("%d tiles in train set, %d tiles in test set, %d files in valid list" % (len(train_list), len(test_list), len(valid_list)))

#Generate the train, test and validation dataset
cloud_features='xyzrgbi'
test_set  = tnt.dataset.ListDataset(test_list,functools.partial(cloud_loader, features_used=cloud_features))
train_set = tnt.dataset.ListDataset(train_list,functools.partial(cloud_loader, features_used=cloud_features))
valid_set = tnt.dataset.ListDataset(valid_list,functools.partial(cloud_loader, features_used=cloud_features))

#%% 3. Definition of the architecture of the PointNet Network

#Definition of the PointNet network for semantic segmentation
def cloud_collate(batch):
    """ Collates a list of dataset samples into a batch list for clouds 
    and a single array for labels
    This function is necessary to implement because the clouds have different sizes (unlike for images)
    """
    clouds, labels = list(zip(*batch))
    labels = torch.cat(labels, 0)
    return clouds, labels

class PointNet(nn.Module):
  """
  PointNet network for semantic segmentation
  """
  
  def __init__(self, MLP_1, MLP_2, MLP_3, n_class=3, input_feat=3, subsample_size = 512, cuda = 1):
    """
    initialization function
    MLP_1, MLP_2 and MLP_3 : int list = width of the layers of 
    multi-layer perceptrons. For example MLP_1 = [32, 64] or [16, 64, 128]
    n_class : int =  the number of class
    input_feat : int = number of input feature
    subsample_size : int = number of points to which the tiles are subsampled
    cuda : int = if 0, run on CPU (slow but easy to debug), if 1 on GPU
    """
    
    super(PointNet, self).__init__() #necessary for all classes extending the module class
    
    self.is_cuda = cuda
    self.subsample_size = subsample_size

    #since we don't know the number of layers in the MLPs, we need to use loops
    #to create the correct number of layers
    
    m1 = MLP_1[-1] #size of the first embeding F1
    m2 = MLP_2[-1] #size of the second embeding F2
    

    #MLP_1: input [input_feat x n] -> f1 [m1 x n]
    modules = []
    for i in range(len(MLP_1)): #loop over the layer of MLP1
      #note: for the first layer, the first in_channels is feature_size 
      modules.append(
          nn.Conv1d(
              in_channels=MLP_1[i-1] if i>0 else input_feat, #trick for single line if-else
              out_channels=MLP_1[i],
              kernel_size=1))
      modules.append(nn.BatchNorm1d(MLP_1[i]))
      modules.append(nn.ReLU(True))
    #this transform the list of layers into a callable module
    self.MLP_1 = nn.Sequential(*modules)

    #MLP_2: f1 [m1 x n] -> f2 [m2 x n]
    modules = []
    for i in range(len(MLP_2)):
      modules.append(nn.Conv1d(MLP_2[i-1] if i>0 else m1, MLP_2[i], 1))
      modules.append(nn.BatchNorm1d(MLP_2[i]))
      modules.append(nn.ReLU(True))
    self.MLP_2 = nn.Sequential(*modules)

    #MLP_3: f1 [(m1 + m2) x n] -> output [k x n]
    modules = []
    for i in range(len(MLP_3)):
      modules.append(nn.Conv1d(MLP_3[i-1] if i>0 else m1 + m2, MLP_3[i], 1))
      modules.append(nn.BatchNorm1d(MLP_3[i]))
      modules.append(nn.ReLU(True))
    #note: the last layer do not have normalization nor activation
    modules.append(nn.Conv1d(MLP_3[-1], n_class,1))
    self.MLP_3 = nn.Sequential(*modules)
    
    self.maxpool = nn.MaxPool1d(subsample_size)
    
    if cuda:
      self = self.cuda()
    
  def forward(self, input):
    """
    the forward function producing the embeddings for each point of 'input'
    input : [n_batch, input_feat, subsample_size] float array = input features
    output : [n_batch,n_class, subsample_size] float array = point class logits
    """
    # print(f"input_size: {input}")
    if self.is_cuda: #put the input on the GPU (at the last moment)
      input = input.cuda()
    #embed points, equation (1)
    # print("feature 1")
    f1 = self.MLP_1(input) 

    # print("feature 2")
    #second point embeddings equation (2)
    f2 = self.MLP_2(f1)

    # print("maxpooling")
    #maxpool, equation 3
    G = self.maxpool(f2)

    # print("concatenation")
    #concatenate f1 and G 
    Gf1 = torch.cat((G.repeat(1,1,self.subsample_size), f1),1)

    # print("global + local featuring")
    #equation(4)
    out = self.MLP_3(Gf1)
    return out

#%% 4. PointNet Architecture Class Working Test

#Parameters setting
#we consider the first point cloud from the training set
features_used = 
cloud_data, gt = 

class_names = 

#to create a proper input for the pointnet we need to add one empty dimension
#for the batch size (with keyword None), and subsample the point cloud to have
#subsample_size = 512 points
#on top, we have to ensure we work with double precision, thus using the .float()
cloud_data = 

#We now create a pointnet model with various parameters:
ptn = 


#we now test that the code works correctly
pred = 

#we now check that the size is indeed [n_batch,n_class, subsample_size]
assert()

#%% 5. Definition of the Semantic Segmentation

#The definition of the Classifier with PointNet
class PointCloudClassifier:
  """
  The main point cloud classifier Class
  deal with subsampling the tiles to a fixed number of points
  and interpolating to the original clouds
  """
  def __init__(self, args):
    self.subsample_size = args.subsample_size #number of points to subsample each point cloud in the batches
    self.n_input_feats = 3 #size of the point descriptors in input
    if 'i' in args.input_feats: #add intensity
      self.n_input_feats += 1
    if 'rgb' in args.input_feats: #add colors
      self.n_input_feats += 3
    self.n_class = args.n_class #number of classes in the prediction
    self.is_cuda = args.cuda #wether to use GPU acceleration
  
  def run(self, model, clouds):
    """
    INPUT:
    model = the neural network
    clouds = list of n_batch tensors of size [n_feat, n_points_i]: batch of 
          n_batch point clouds of size n_points_i
    OUTPUT:
    pred = [sum_i n_points_i, n_class] float tensor : prediction for each element of the 
         batch concatenated in a single tensor
    """
    #number of batch
    n_batch = len(clouds)

    #will contain the prediction for all clouds in the batch (the output)
    prediction_batch   = torch.zeros((self.n_class,0))
    
    #sampled_clouds contain the clouds from the batch, each subsampled to self.subsample_size points
    sampled_clouds = torch.Tensor(n_batch, self.n_input_feats, self.subsample_size)
    if self.is_cuda:
      prediction_batch = prediction_batch.cuda()

    for i_batch in range(n_batch):
      #load the elements in the batch one by one and subsample/ oversample them
      #to a size of self.subsample_size points
            
      cloud = clouds[i_batch][:,:] #the full cloud
      
      n_points = cloud.shape[1]  # number of points in this cloud   

      selected_points = np.random.choice(n_points, self.subsample_size)
      sampled_cloud = cloud[:,selected_points] #reduce the current cloud to the selected points
        
      sampled_clouds[i_batch,:,:] = sampled_cloud #add the sample cloud to sampled_clouds
    
    #we now have a tensor containing a batch of clouds with the same size
    sampled_prediction = model(sampled_clouds) #classify the batch of sampled clouds
    
    #interpolation to the original point clouds
    for i_batch in range(n_batch):
      #the original point clouds (only xyz position)
      cloud = clouds[i_batch][:3,:]
      #and the corresponding sampled batch (only xyz position)
      sampled_cloud = sampled_clouds[i_batch,:3,:]      
      
      #we now interpolate the prediction of the points of "sampled_cloud" to the original point cloud "cloud"
      #with nearest neighbor interpolation
      knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit( \
             sampled_cloud.cpu().permute(1,0))
      #select for each point in the original point cloud the closest point in sampled_cloud
      dump, closest_point = knn.kneighbors(cloud.permute(1,0).cpu())
      #remove uneeded dimension (we only took one neighbor)
      closest_point = closest_point.squeeze()

      #prediction for the original point cloud i_batch-> 
      #each point in the original point cloud get the label of the closest point 
      #in the sampled_cloud
      prediction_full_cloud = sampled_prediction[i_batch,:,closest_point]
    
      #we append prediction_full_cloud to prediction_batch
      prediction_batch = torch.cat((prediction_batch, prediction_full_cloud),1)
     
    return prediction_batch.permute(1,0)

#%% 6. Code Working Test
"""
This is to verify that we are able to use the PointNet Classifier and different point clouds
"""

#We first consider the point cloud from the training set
features_used = 
cloud1 = 
cloud2 = 

#we artificially decrease the number of points in cloud2 in order to test both 
#subample and oversample cases
cloud2 = 
print(f"Loading two clouds with {cloud1[0].shape[1]}  and {cloud2[0].shape[1]} points respectively" )

batch_clouds, gt = 

#We define the arguments used for initializing our instance
args = 
args.n_class = 
args.input_feats = 
args.subsample_size = 
args.cuda = 

#we create an instance of PointCloudClassifier
PCC = 

#we create a PointNet model
model = 
model = 

#we now launch the prediction:
pred = 

#we check that the size of the prediction is indeed [sum_i, n_points_i,n_class]
assert()

#%% 7. Definition of the Metrics

#Definition of the confusion matrix
class ConfusionMatrix:
  def __init__(self, n_class, class_names):
    self.CM = np.zeros((n_class, n_class))
    self.n_class = n_class
    self.class_names = class_names
  
  def clear(self):
    self.CM = np.zeros((self.n_class, self.n_class))
    
  def add_batch(self, gt, pred):
    self.CM +=  confusion_matrix(gt, pred, labels = list(range(self.n_class)))
    
  def overall_accuracy(self):
    return 100*self.CM.trace() / self.CM.sum()

  def class_IoU(self, show = 1):
    ious = np.diag(self.CM)/ (np.sum(self.CM,1)+np.sum(self.CM,0)-np.diag(self.CM))
    if show:
      print('  |  '.join('{} : {:3.2f}%'.format(name, 100*iou) for name, iou in zip(self.class_names,ious)))

    return 100*np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()

#%% 8. Definition of Training Functions of PointNet

def train(model, PCC, optimizer, args):
  """train for one epoch"""
  model.train()
  
  #the loader function will take care of the batching
  loader = torch.utils.data.DataLoader(train_set, collate_fn=cloud_collate, \
         batch_size=args.batch_size, shuffle=True, drop_last=True)
  #tqdm will provide some nice progress bars
  loader = tqdm(loader, ncols=100, leave=False, desc="Train Epoch")
  
  #will keep track of the loss
  loss_meter = tnt.meter.AverageValueMeter()
  cm = ConfusionMatrix(args.n_class, class_names = class_names[1:])

  for index_batch, (cloud, gt) in enumerate(loader):

    if PCC.is_cuda:
      gt = gt.cuda()
     
    optimizer.zero_grad() #put gradient to zero
    
    pred = PCC.run(model, cloud) #compute the prediction

    labeled = gt!=0 #remove the unlabelled points from the supervision
    if labeled.sum() == 0:
      continue #no labeled points, skip
    
    loss = nn.functional.cross_entropy(pred[labeled], gt[labeled]-1) #-1 to account for the unlabelled class

    loss.backward() #compute gradients
    
    optimizer.step() #one SGD step
    
    loss_meter.add(loss.item())
    #we need to convert back to numpy array, which requires detaching gradients and putting back in RAM
    cm.add_batch(gt[labeled].cpu()-1, pred[labeled].argmax(1).cpu().detach().numpy()) 
  return cm, loss_meter.value()[0]

def eval(model, PCC, test, args):
  """eval on test/valid set"""
  
  model.eval()
  
  if test: #evaluate on test set
    loader = torch.utils.data.DataLoader(test_set, collate_fn=cloud_collate, batch_size=args.batch_size, shuffle=False)
    loader = tqdm(loader, ncols=500, leave=False, desc="Test")
  else: #evaluate on valid set
    loader = torch.utils.data.DataLoader(valid_set, collate_fn=cloud_collate, batch_size=60, shuffle=False, drop_last=False)
    loader = tqdm(loader, ncols=500, leave=False, desc="Val")
  
  loss_meter = tnt.meter.AverageValueMeter()
  cm = ConfusionMatrix(args.n_class, class_names=class_names[1:])

  for index_batch, (cloud, gt) in enumerate(loader):
    #like train, without gradients
    if PCC.is_cuda:
      gt = gt.cuda()
    with torch.no_grad():
      pred = PCC.run(model, cloud)  
      labeled = gt!=0 #we remove the unlabelled points from the supervision
      if labeled.sum() == 0:
        continue #no labeled points, skip
      loss = nn.functional.cross_entropy(pred[labeled], gt[labeled]-1) 
      loss_meter.add(loss.item())
      cm.add_batch((gt[labeled]-1).cpu(), pred[labeled].argmax(1).cpu().detach().numpy())

  return cm, loss_meter.value()[0]

def train_full(args):
  """The full training loop"""
  #initialize the model
  model = PointNet(args.MLP_1, args.MLP_2, args.MLP_3, args.n_class, input_feat=args.n_input_feats, subsample_size = args.subsample_size)

  print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
  # print(model)
  
  best_model = None
  best_mIoU = 0

  #define the classifier
  PCC = PointCloudClassifier(args)
  
  #define the optimizer
  #adam optimizer is always a good guess for classification
  optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
  #adding a scheduler for adaptive learning rate. at each milestone lr_steps 
  
  TESTCOLOR = '\033[104m' #color for test
  TRAINCOLOR = '\033[100m' #color for train
  VALIDCOLOR = '\033[45m' #color for validation
  NORMALCOLOR = '\033[0m'
  
  metrics_pn={}
  metrics_pn['definition']=[['train_oa','train_mIoU','train_loss'],['valid_oa','valid_mIoU','valid_loss'],['test_oa','test_mIoU','test_loss']]
  for i_epoch in tqdm(range(args.n_epoch), desc="Training"):
    #train one epoch
    cm_train, loss_train = train(model, PCC, optimizer, args)
    mIoU = cm_train.class_IoU()
    tqdm.write(TRAINCOLOR + 'Epoch %3d -> Train Overall Accuracy: %3.2f%% Train mIoU : %3.2f%% Train Loss: %1.4f' % (i_epoch, cm_train.overall_accuracy(), mIoU, loss_train) + NORMALCOLOR)
    metrics_pn[i_epoch]=[[cm_train.overall_accuracy(), mIoU, loss_train]]
    
    cm_valid, loss_valid = eval(model, PCC, False, args) #evaluate on validation set
    mIoU_valid = cm_valid.class_IoU()
    metrics_pn[i_epoch].append([cm_valid.overall_accuracy(), mIoU_valid, loss_valid])
    
    best_valid = False
    if mIoU_valid > best_mIoU: #we have reached a new best value on the valdiation set
      best_valid = True #boolean indicating that we have reached a new best value
      best_mIoU = mIoU_valid
      
      best_model = copy.deepcopy(model)#we copy and store the model
      tqdm.write(VALIDCOLOR + '-> Best performance achieved: Overall Accuracy: %3.2f%% valid mIoU : %3.2f%% valid Loss: %1.4f' % (cm_valid.overall_accuracy(), best_mIoU, loss_valid) + NORMALCOLOR)
    else:
      tqdm.write(NORMALCOLOR + 'Subpar performance achieved: Overall Accuracy: %3.2f%% valid mIoU : %3.2f%% valid Loss: %1.4f' % (cm_valid.overall_accuracy(), best_mIoU, loss_valid) + NORMALCOLOR)

    if i_epoch == args.n_epoch-1 or best_valid :
      #evaluate on test set
      cm_test, loss_test = eval(best_model, PCC, True, args)
      mIoU = cm_test.class_IoU()
      tqdm.write(TESTCOLOR + 'Test Overall Accuracy: %3.2f%% Test mIoU : %3.2f%%  Test Loss: %1.4f' % (cm_test.overall_accuracy(), mIoU, loss_test) + NORMALCOLOR)
      metrics_pn[i_epoch].append([cm_test.overall_accuracy(), mIoU, loss_test])
      
  return best_model, PCC, metrics_pn

#%% 9. Definition of the parameters

#structure where we store arguments
args = 

#arguments to experiment on
class_names = 
args.n_epoch = 
args.subsample_size = 

#leave these arguments unchanged
args.n_epoch_test = 
args.batch_size = 
args.n_class = 
args.input_feats = 
args.n_input_feats = 
args.MLP_1 = 
args.MLP_2 = 
args.MLP_3 = 
args.show_test = 
args.lr = 
args.wd = 
args.cuda = 

#%% 10. PointNet Training

# ~ 60 minutes for 30 epochs and 512 subsample size, 
#Training the model
t0 = time.time()
trained_model, PCC, metrics = train_full(args)
t1 = time.time()

print(trained_model)
print(f"{'-'*50}")
print(f"Total training time: {t1-t0} seconds")
print(f"{'-'*50}")

#%% 11. PointNet Prediction Vizualisation
def tile_prediction(tile_name, model=None, PCC=None, Visualization=True, features_used='xyzrgbi'):
    
    # Load the tile 
    cloud, gt = cloud_loader(tile_name, features_used)
    
    #Make the predictions
    labels = PCC.run(model, [cloud])
    labels = labels.argmax(1).cpu() + 1

    #Prepare the data for export
    xyz = np.array(cloud[0:3]).transpose()
    
    #Prepare the data for open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    #Vizualisation with Open3d
    if Visualization==True:
        max_label = labels.max()
        colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        pcd.estimate_normals(fast_normal_computation=True)
        o3d.visualization.draw_geometries([pcd])
    
    return pcd, labels

selection = test_list[9]
pcd, labels = tile_prediction(selection, model=trained_model, PCC=PCC)

#%% 12. PointNet Model Export

#saving the model weights
torch.save(trained_model.state_dict(), "../pointnet_course_"+project_dir.split("DATA/")[1]+".torch")

#saving the metrics
with open('../metrics_'+project_dir.split("DATA/")[1]+'.csv', "w") as f:
    for key, value in metrics.items(): 
        f.write('%s:%s\n' % (key, value))