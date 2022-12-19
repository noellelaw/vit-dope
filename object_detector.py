# Object detector code for inference
# Slightly adapted from https://github.com/NVlabs/Deep_Object_Pose

#---------------------------------------------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------------------------------------------
import copy
import os
import os.path as osp
from os.path import exists
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import time
import warnings
import numpy as np
import json
import datetime
import glob
import cv2
import colorsys
import math

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

from PIL import Image
from PIL import ImageDraw
from PIL import ImageEnhance

from math import acos
from math import sqrt
from math import pi  

import mmcv
from mmcv import Config, DictAction
from mmcv.utils import get_git_hash
from mmcv.runner import get_dist_info, init_dist, set_random_seed

from collections import OrderedDict
import tempfile
import random
from __future__ import print_function

from models.backbones import ViT
from core.evaluation.top_down_eval import (keypoint_pck_accuracy,
                            keypoints_from_heatmaps,
                            pose_pck_accuracy)
from models.heads import TopdownHeatmapSimpleHead
#---------------------------------------------------------------------------------------------------------------
# Make a grid of images for testing purposes
#---------------------------------------------------------------------------------------------------------------
def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range_=None, scale_each=False, pad_value=0):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range_ is not None:
            assert isinstance(range_, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range_):
            if range_ is not None:
                norm_ip(t, range_[0], range_[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(tensor[k])
            k = k + 1
    return grid


#---------------------------------------------------------------------------------------------------------------
# Get grid of images
#---------------------------------------------------------------------------------------------------------------
def get_image_grid(tensor, nrow=3, padding=2, mean=None, std=None):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    from PIL import Image

    # tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=1)
    if not mean is None:
        # ndarr = grid.mul(std).add(mean).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
        ndarr = grid.mul(std).add(mean).mul(255).byte().transpose(0, 2).transpose(0, 1).numpy()
    else:
        ndarr = grid.mul(0.5).add(0.5).mul(255).byte().transpose(0, 2).transpose(0, 1).numpy()
    im = Image.fromarray(ndarr)
    return im

#---------------------------------------------------------------------------------------------------------------
# OBJECT DETECTOR CLASS 
#---------------------------------------------------------------------------------------------------------------
class ObjectDetector(object):
    '''This class contains methods for object detection'''

    @staticmethod
    def find_object_poses(vertex, aff):
        '''Detect objects given network output'''

        # Detect objects from belief maps and affinities
        objects, all_peaks = ObjectDetector.find_objects(vertex, aff)
        detected_objects = []
        obj_name = 'mustard'

        for obj in objects:
            points = obj[1] + [(obj[0][0]*8, obj[0][1]*8)]
            cuboid2d = np.copy(points)

            # Save results
            detected_objects.append({
                'name': obj_name,
                'cuboid2d': cuboid2d,
                'score': obj[-1],
            })
        print(detected_objects)
        return detected_objects

    @staticmethod
    def find_objects(vertex, aff, numvertex=8):
        '''Detects objects given network belief maps and affinities, using heuristic method'''

        all_peaks = []
        peak_counter = 0
        for j in range(vertex.size()[0]):
            belief = vertex[j].clone()
            map_ori = belief.cpu().data.numpy()

            map = gaussian_filter(belief.cpu().data.numpy(), sigma=sigma)
            print(map.shape)
            p = 1
            map_left = np.zeros(map.shape)
            map_left[p:,:] = map[:-p,:]
            map_right = np.zeros(map.shape)
            map_right[:-p,:] = map[p:,:]
            map_up = np.zeros(map.shape)
            map_up[:,p:] = map[:,:-p]
            map_down = np.zeros(map.shape)
            map_down[:,:-p] = map[:,p:]

            peaks_binary = np.logical_and.reduce(
                                (
                                    map >= map_left,
                                    map >= map_right,
                                    map >= map_up,
                                    map >= map_down,
                                    map > thresh_map)
                                )
            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])

            # Computing the weigthed average for localizing the peaks
            peaks = list(peaks)
            win = 5
            ran = win // 2
            peaks_avg = []
            for p_value in range(len(peaks)):
                p = peaks[p_value]
                weights = np.zeros((win,win))
                i_values = np.zeros((win,win))
                j_values = np.zeros((win,win))
                for i in range(-ran,ran+1):
                    for j in range(-ran,ran+1):
                        if p[1]+i < 0 \
                                or p[1]+i >= map_ori.shape[0] \
                                or p[0]+j < 0 \
                                or p[0]+j >= map_ori.shape[1]:
                            continue

                        i_values[j+ran, i+ran] = p[1] + i
                        j_values[j+ran, i+ran] = p[0] + j

                        weights[j+ran, i+ran] = (map_ori[p[1]+i, p[0]+j])

                # if the weights are all zeros
                # then add the none continuous points
                OFFSET_DUE_TO_UPSAMPLING = 0.4395
                try:
                    peaks_avg.append(
                        (np.average(j_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING, \
                         np.average(i_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING))
                except:
                    peaks_avg.append((p[0] + OFFSET_DUE_TO_UPSAMPLING, p[1] + OFFSET_DUE_TO_UPSAMPLING))
            # Note: Python3 doesn't support len for zip object
            peaks_len = min(len(np.nonzero(peaks_binary)[1]), len(np.nonzero(peaks_binary)[0]))

            peaks_with_score = [peaks_avg[x_] + (map_ori[peaks[x_][1],peaks[x_][0]],) for x_ in range(len(peaks))]

            id = range(peak_counter, peak_counter + peaks_len)

            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += peaks_len

        objects = []

        # Check object centroid and build the objects if the centroid is found
        for nb_object in range(len(all_peaks[-1])):
            if all_peaks[-1][nb_object][2] > thresh_points:
                objects.append([
                    [all_peaks[-1][nb_object][:2][0],all_peaks[-1][nb_object][:2][1]],
                    [None for i in range(numvertex)],
                    [None for i in range(numvertex)],
                    all_peaks[-1][nb_object][2]
                ])

        # Working with an output that only has belief maps
        if aff is None:
            if len (objects) > 0 and len(all_peaks)>0 and len(all_peaks[0])>0:
                for i_points in range(8):
                    if  len(all_peaks[i_points])>0 and all_peaks[i_points][0][2] > threshold:
                        objects[0][1][i_points] = (all_peaks[i_points][0][0], all_peaks[i_points][0][1])
        else:
            # For all points found
            for i_lists in range(len(all_peaks[:-1])):
                lists = all_peaks[i_lists]

                for candidate in lists:
                    if candidate[2] < thresh_points:
                        continue

                    i_best = -1
                    best_dist = 10000
                    best_angle = 100
                    for i_obj in range(len(objects)):
                        center = [objects[i_obj][0][0], objects[i_obj][0][1]]

                        # integer is used to look into the affinity map,
                        # but the float version is used to run
                        point_int = [int(candidate[0]), int(candidate[1])]
                        point = [candidate[0], candidate[1]]

                        # look at the distance to the vector field.
                        v_aff = np.array([
                                        aff[i_lists*2,
                                        point_int[1],
                                        point_int[0]].data.item(),
                                        aff[i_lists*2+1,
                                            point_int[1],
                                            point_int[0]].data.item()]) * 10

                        # normalize the vector
                        xvec = v_aff[0]
                        yvec = v_aff[1]

                        norms = np.sqrt(xvec * xvec + yvec * yvec)

                        xvec/=norms
                        yvec/=norms

                        v_aff = np.concatenate([[xvec],[yvec]])

                        v_center = np.array(center) - np.array(point)
                        xvec = v_center[0]
                        yvec = v_center[1]

                        norms = np.sqrt(xvec * xvec + yvec * yvec)

                        xvec /= norms
                        yvec /= norms

                        v_center = np.concatenate([[xvec],[yvec]])

                        # vector affinity
                        dist_angle = np.linalg.norm(v_center - v_aff)

                        # distance between vertexes
                        dist_point = np.linalg.norm(np.array(point) - np.array(center))

                        if dist_angle < thresh_angle and (best_dist > 1000 or best_dist > dist_point):
                            i_best = i_obj
                            best_angle = dist_angle
                            best_dist = dist_point

                    if i_best == -1:
                        continue

                    if objects[i_best][1][i_lists] is None \
                            or best_angle < thresh_angle \
                            and best_dist < objects[i_best][2][i_lists][1]:
                        objects[i_best][1][i_lists] = ((candidate[0])*8, (candidate[1])*8)
                        objects[i_best][2][i_lists] = (best_angle, best_dist)

        return objects, all_peaks