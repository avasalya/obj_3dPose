#!/usr/bin/env python

import time
import numpy as np
# import pcl
from matplotlib import pyplot
from numpy import *
import open3d as o3d


def PreprocessPointCloud(pc, bbox, height=None, width=None, limits=None):
    if height is None or width is None:
            raise Exception("PointCloud height/width undefined")
    # crop pcd data with bbox
    # print ("pc1",pc)

    Y, X = np.meshgrid(np.arange(bbox[1], bbox[3]), np.arange(bbox[0], bbox[2]))
    indices = np.ravel_multi_index((Y.flatten(), X.flatten()), (height, width))

    # print ("indices", indices)
    pc_crop_xy = pc.select_down_sample(indices, invert=False) #open3d
    # print ("limit", limits)
    points = np.asarray(pc_crop_xy.points)
    colors = np.asarray(pc_crop_xy.colors)
    pc_data = np.c_[points, colors]
    d = list()
    for element in pc_data:
        if (element[0] != float("nan")) and (element[2]>=limits[4] and element[2]<=limits[5]):
            d.append(element)

    d = np.array(d)
    pc_crop_xyz = o3d.geometry.PointCloud()
    pc_crop_xyz.points = o3d.utility.Vector3dVector(d[:, 0:3])
    pc_crop_xyz.colors = o3d.utility.Vector3dVector(d[:, 3:6])



    #TODO - cylinder cut
    # # remove support plane conservatively
    # seg = pc_filtered.make_segmenter_normals(ksearch=50)
    # seg.set_optimize_coefficients(True)
    # seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    # seg.set_normal_distance_weight(0.1)
    # seg.set_method_type(pcl.SAC_RANSAC)
    # seg.set_max_iterations(100)
    # seg.set_distance_threshold(0.03)
    # indices, model = seg.segment()
    # pc_plane = pc_filtered.extract(indices, negative=False)

    # pc_cylinder = pc_filtered.extract(indices, negative=True)

    # '''
    # seg = pc_cylinder.make_segmenter_normals(ksearch=50)
    # seg.set_optimize_coefficients(True)
    # seg.set_model_type(pcl.SACMODEL_CYLINDER)
    # seg.set_normal_distance_weight(0.05)
    # seg.set_method_type(pcl.SAC_RANSAC)
    # seg.set_max_iterations(1000)
    # seg.set_distance_threshold(0.1)
    # seg.set_radius_limits(0,0.2)
    
    # indices, model = seg.segment()
    # pc_cylinder = pc_cylinder.extract(indices, negative=False)pc_cylinder 
    # '''
    


    return pc_crop_xyz 
