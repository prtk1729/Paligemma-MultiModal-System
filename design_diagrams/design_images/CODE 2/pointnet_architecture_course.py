# -*- coding: utf-8 -*-
"""
3D Deep Learning: Semantic Segmentation
PointNet 2/4 - Model Architecture
(c) Florent Poux, 2023
Licence: Prorpietary.
Members of the 3D Deep Learning Course are given Explicit consent for unrestricted commercial applications.
Credits and attributions are mandatory in both cases.
"""

#Base Math libraries
import numpy as np

#Deep Learning Libraries
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

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
