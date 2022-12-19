# --------------------------------------------------------------------------------------------
# ViTDope training code
# Adapted from https://github.com/NVlabs/Deep_Object_Pose
#---------------------------------------------------------------------------------------------------------------
# IMPORTS
#---------------------------------------------------------------------------------------------------------------
import copy
import os
import os.path as osp
from os.path import exists
import time
import warnings
import numpy as np
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.models as models
from torch.distributions import MultivariateNormal as MVN
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import cross_entropy

import mmcv 

import tempfile
import random
from __future__ import print_function

from models.backbones import ViT
from core.evaluation.top_down_eval import (keypoint_pck_accuracy,
                            keypoints_from_heatmaps,
                            pose_pck_accuracy)
from models.heads import TopdownHeatmapSimpleHead

#---------------------------------------------------------------------------------------------------------------
# HYPERPARAMETERS
# TODO:
# - Switch to argument parser or config file
#---------------------------------------------------------------------------------------------------------------
DATA_PATH = ''
DATA_PATH_TEST = ''
PRETRAINED = '' # Put path to ViT MAE pretaining weights here
FROM_NET = ''
YCB_OBJECT = 'cracker_box'
NOISE = 1e-5
BRIGHTNESS = 1e-5
CONTRAST = 1e-5
BATCH_SIZE = 2
IMAGE_SIZE = 256
LEARNING_RATE = 5e-4
EPOCHS = 60
LOG_INTERVAL = 100
SIGMA = 4
OUT_FLDR = '/content/drive/MyDrive/DeepLearning/cracker_box_trainone
SAVE = False
NORMAL_IMGS = None
NB_UPDATES = None
NAME_FILE = 'epoch'
MAX_NORM = 1.
NORM_TYPE = 2
# For Balanced MSE
BMCE_LOSS = 0.1
# For inference
thresh_angle=0.5
thresh_map=0.01
sigma=3
thresh_points=0.1

#---------------------------------------------------------------------------------------------------------------
# Image Transform
#---------------------------------------------------------------------------------------------------------------
transform = transforms.Compose([
                          transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                          transforms.ToTensor()])

#---------------------------------------------------------------------------------------------------------------
# ViTDope
#---------------------------------------------------------------------------------------------------------------
class ViTDopeNetwork(nn.Module):
  def __init__(
            self,
            pretrained=False,
            numBeliefMap=9,
            numAffinity=16
            ):
    super(ViTDopeNetwork, self).__init__()
    # Set up backbone accordance with ViT-B
    backbone = ViT(img_size=(256,256),
                  patch_size=16,
                  embed_dim=768,
                  depth=12,
                  num_heads=12,
                  ratio=1,
                  use_checkpoint=False,
                  mlp_ratio=4,
                  qkv_bias=True,
                  drop_path_rate=0.3,
    )
    # Init ViT weights from ViT MAE trained on image net
    if not PRETRAINED == '':
        backbone.init_weights(pretrained=PRETRAINED)
    # Set classical decoder head for belief maps
    belief_head = TopdownHeatmapSimpleHead(
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=numBeliefMap,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)
    )
    # Set classical decoder head for affity maps
    affinity_head = TopdownHeatmapSimpleHead(
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=numAffinity,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)
    )

    self.backbone = nn.Sequential(*[backbone])
    self.belief_head = nn.Sequential(*[belief_head])
    self.affinity_head = nn.Sequential(*[affinity_head])

  def forward(self, x):
    backbone_out = self.backbone(x)
    belief_out = self.belief_head(backbone_out)
    affinity_out = self.affinity_head(backbone_out)
    return belief_out, affinity_out

#---------------------------------------------------------------------------------------------------------------
# Get balanced mce loss
#---------------------------------------------------------------------------------------------------------------
def get_bmce_loss(preds, targets):
    B,N,H,W = preds.shape # Batch size, num outputs, height, width
    resize_to = H*W
    loss = 0
    # Is there a more readable way to code this?
    for i in range(N):         
        I = torch.eye(H*W)
        belief = preds[:,i,:,:].reshape( (B,resize_to) ).cpu()
        target = targets[:,i,:,:].reshape( (B,resize_to) ).cpu() # logit size: [batch, batch]
        logits = MVN( belief.unsqueeze(1), (BMCE_LOSS*I) ).log_prob( target.unsqueeze(0) )  
        loss_temp = cross_entropy(logits, torch.arange(B))  # contrastive-like loss
        loss_temp = loss_temp * (2 * BMCE_LOSS)
        loss += loss_temp
    return loss

#---------------------------------------------------------------------------------------------------------------
# Run the network for one epoch 
#---------------------------------------------------------------------------------------------------------------
def _run_network(epoch, loader, train=True):

    if train:
        net.train()
    else:
        net.eval()

    # Iterate through batches
    for batch_idx, targets in enumerate(loader):
        # Get data and targets
        data = Variable(targets['img'].cuda())
        target_belief = Variable(targets['beliefs'].cuda())        
        target_affinity = Variable(targets['affinities'].cuda())
        loss = None
        if train:
            optimizer.zero_grad()

        # Get predictions
        output_belief, output_affinities = net(data) 

        # Belief maps loss
        for l in output_belief: #output, each belief map layers. 
            if loss is None:
                loss = ((l - target_belief) * (l-target_belief)).mean()
                
            else:
                loss_tmp = ((l - target_belief) * (l-target_belief)).mean()
                loss += loss_tmp
        # Get balanced mce loss for belief maps        
        loss += get_bmce_loss(output_belief, target_belief)

        # Affinities loss
        for l in output_affinities: #output, each belief map layers. 
            loss_tmp = ((l - target_affinity) * (l-target_affinity)).mean()
            loss += loss_tmp 
        # Get balanced mce loss for belief maps  
        loss += get_bmce_loss(output_affinities, target_affinity)

        # Update weights
        if train:
            loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(parameters, max_norm=MAX_NORM, norm_type=NORM_TYPE)
            optimizer.step()

        # Determine file to write loss into 
        if train:
            namefile = '/loss_train.csv'
        else:
            namefile = '/loss_test.csv'
        # Write to files
        with open (OUT_FLDR+namefile,'a') as file:
            s = '{}, {},{:.15f}\n'.format(
                epoch,batch_idx,loss.data.item()) 
            file.write(s)

        # Print results
        if train:
            if batch_idx % LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset),
                    100. * batch_idx / len(loader), loss.data.item()))
        else:
            if batch_idx % LOG_INTERVAL == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset),
                    100. * batch_idx / len(loader), loss.data.item()))

#---------------------------------------------------------------------------------------------------------------
# Get training dataloader
#---------------------------------------------------------------------------------------------------------------
print ("Loading data...")
trainingdata = None
if not DATA_PATH == "":
  train_dataset = MultipleVertexJson(
      root=DATA_PATH,
      objectofinterest=YCB_OBJECT,
      keep_orientation = True,
      noise = NOISE,
      sigma = SIGMA,
      data_size = DATASIZE,
      save = SAVE,
      transform = transform,
      normal = NORMAL_IMGS,
      target_transform = transforms.Compose([
                              transforms.Resize(IMAGE_SIZE//4),
          ]),
      )
  trainingdata = torch.utils.data.DataLoader(train_dataset,
      batch_size = BATCH_SIZE, 
      shuffle = True,
      num_workers = 1, 
      pin_memory = True
      )

#---------------------------------------------------------------------------------------------------------------
# Get testing dataloader
#---------------------------------------------------------------------------------------------------------------testingdata = None
if not DATA_PATH_TEST == "":
  testingdata = torch.utils.data.DataLoader(
      MultipleVertexJson(
          root = DATA_PATH_TEST,
          objectofinterest=YCB_OBJECT,
          keep_orientation = True,
          noise = NOISE,
          sigma = SIGMA,
          data_size = DATASIZE,
          save = SAVE,
          transform = transform,
          normal = NORMAL_IMGS,
          target_transform = transforms.Compose([
                                  transforms.Resize(IMAGE_SIZE//4),
              ]),
          ),
      batch_size = BATCH_SIZE, 
      shuffle = True,
      num_workers = 1, 
      pin_memory = True)

#---------------------------------------------------------------------------------------------------------------
# Get model and optimizer
#---------------------------------------------------------------------------------------------------------------
print("Loading model...")
net = ViTDopeNetwork()
net = net.to('cuda')
# Load to resume training
if FROM_NET!= '':
    net.load_state_dict(torch.load(FROM_NET))
# Set up optimizer and scheduler
parameters = filter(lambda p: p.requires_grad, net.parameters())
optimizer = torch.optim.AdamW(parameters,
                              lr=LEARNING_RATE, 
                              betas=(0.9, 0.999), 
                              weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=10)

#---------------------------------------------------------------------------------------------------------------
# Write to files
#---------------------------------------------------------------------------------------------------------------
with open (OUT_FLDR+'/loss_train.csv','w') as file: 
    file.write('epoch,batchid,loss\n')

with open (OUT_FLDR+'/loss_test.csv','w') as file: 
    file.write('epoch,batchid,loss\n')

#---------------------------------------------------------------------------------------------------------------
# Start training
#---------------------------------------------------------------------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    # Run training
    if not trainingdata is None:
        _run_network(epoch,trainingdata)

    if not DATA_PATH_TEST == "":
        _run_network(epoch,testingdata,train = False)
        if  DATA_PATH == "":
            break # lets get out of this if we are only testing
    try:
        # Save weights 
        torch.save(net.state_dict(), '{}/net_{}_{}.pth'.format(OUT_FLDR, NAME_FILE, epoch))
    except:
        pass
    
    scheduler.step()

print ("end:" , datetime.datetime.now().time())