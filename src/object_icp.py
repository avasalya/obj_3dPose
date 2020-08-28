#!/usr/bin/env python

'''
iterative_closest_ponint2.py

This code performs a sequence of operations (SSD (detection) -> PoseCNN -> ICP)
to automatically detect the bounding box and pose of the object from an RGBD image.
It will visualize the cropped image and the regstgration result where the cad model is alined to the point cloud.

You can test RGBD images in "dataset/tool20170123_rcnn".

python iterative_closest_ponint.py #image

'''
import sys
# sys.path.append('/home/ash/catkin_ws/devel/lib/python3/dist-packages') # in order to import cv2 under python3
# sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages') # append back in order to import rospy

import time
import numpy as np

# import pcl
import open3d.open3d as o3d
import torch
import torch.nn as nn
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy import *
from torch.autograd import Variable
from torchvision import models
from sys import argv
import tools.Geometry as geo
import tools.Transform as tf
import tools.readOff as readOff
import tools.ImageDataConvert as idc

import cv2
from scipy.spatial import cKDTree


def PreprocessPointCloud(pcFile, bbox, height=None, width=None, limits=None):
    if type(pcFile) == str:

        pc = pcl.load_XYZRGB(pcFile) # point cloud
        # pcd = o3d.io.read_point_cloud("pcFile", format='xyzrgb')



        height = pc.height
        width = pc.width
        print(height)
        print(width)

    elif type(pcFile) == pcl._pcl.PointCloud_PointXYZRGB or type(pcFile) == pcl._pcl.PointCloud:
        pc = pcFile
        if height is None or width is None:
            raise Exception("PointCloud height/width undefined")

    # crop pcd data with bbox
    Y, X = np.meshgrid(np.arange(bbox[1], bbox[3]), np.arange(bbox[0], bbox[2]))
    indices = np.ravel_multi_index((Y.flatten(), X.flatten()), (height, width))
    pc = pc.extract(indices, negative=False)


    # +++++++++++++++++++++++++++++ segment point cloud +++++++++++++++++++++++++++++

    # c.f. https://github.com/strawlab/python-pcl/blob/master/examples/segment_cyl_plane.py



    fil = pc.make_passthrough_filter()

    fil.set_filter_field_name("x")
    fil.set_filter_limits(limits[0],limits[1] )# depth 0 m <=> 3 m
    
    fil.set_filter_field_name("y")
    fil.set_filter_limits(limits[2],limits[3])# depth 0 m <=> 3 m
    
    fil.set_filter_field_name("z")
    fil.set_filter_limits(limits[4],limits[5])# depth 0 m <=> 3 m
    pc_filtered = fil.filter()
    pc_crop = np.asarray(pc_filtered)[:, 0:4]


    # remove support plane conservatively
    seg = pc_filtered.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_normal_distance_weight(0.1)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(100)
    seg.set_distance_threshold(0.03)
    indices, model = seg.segment()
    pc_plane = pc_filtered.extract(indices, negative=False)

    pc_cylinder = pc_filtered.extract(indices, negative=True)

    '''
    seg = pc_cylinder.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_CYLINDER)
    seg.set_normal_distance_weight(0.05)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(1000)
    seg.set_distance_threshold(0.1)
    seg.set_radius_limits(0,0.2)
    
    indices, model = seg.segment()
    pc_cylinder = pc_cylinder.extract(indices, negative=False)
    '''
    


    return pc_crop, pc_cylinder


def VisualizeICPResult(imageFile, bbox, model, model_init, target, pc_raw):

    I = pyplot.imread(imageFile)
    I_crop = I[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    Rc = np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    #Rc = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    fig1 = pyplot.figure(1)
    rect = pyplot.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fc="none", ec="red")
    ax1 = fig1.add_subplot(111)
    ax1.add_patch(rect)
    pyplot.imshow(I)
    pyplot.figure(2)
    pyplot.imshow(I_crop)


    model_init = dot(Rc, model_init.T).T
    target.vertex = dot(Rc, target.vertex.T).T
    model.vertex = dot(Rc, model.vertex.T).T
    pc_raw = dot(Rc, pc_raw.T).T
    model.centroid = dot(Rc, model.centroid.T).T

    fig = pyplot.figure(3, figsize=(8, 8))
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(model_init[:, 0], model_init[:, 1], model_init[:, 2], triangles=F, linewidth=0.2, antialiased=True,
                    color='lightblue', alpha=0.7)
    ax.plot(target.vertex[:, 0], target.vertex[:, 1], target.vertex[:, 2], ".", color="yellow", ms=0.5, mew=1)
    ax.plot(pc_raw[:, 0], pc_raw[:, 1], pc_raw[:, 2], ".", color="orange", ms=0.3, mew=1, alpha=0.5)
    ax.plot_trisurf(model.vertex[:, 0], model.vertex[:, 1], model.vertex[:, 2], triangles=F, linewidth=0.2,
                    antialiased=True, color='blue')
    ax.plot(np.asarray([model.centroid[0], model.centroid[0]]), np.asarray([model.centroid[1], model.centroid[1]]),
            np.asarray([model.centroid[2], model.centroid[2]]), "o", color="red", ms=3, mew=1)
    readOff.axisEqual3D(ax)
    ax.set_aspect('equal')
    ax.view_init(elev=0, azim=250)
    ax.set_label("x - axis")
    ax.set_label("y - axis")
    ax.set_label("z - axis")
    pyplot.show()


class PosePredictor(object):
    
    def __init__(self, pose_cnn_param_file, device):


        
        if pose_cnn_param_file.split('.')[1] == 'alexnet':

            self.net = models.alexnet(pretrained=True)
            self.device = device
            in_feats = self.net.classifier._modules['6'].in_features
            self.net.classifier._modules['6'] = nn.Linear(in_feats, 9)

        elif pose_cnn_param_file.split('.')[1] == 'resnet':
        
        
            self.net = models.resnet50(pretrained=True)
            self.device = device
            in_feats = self.net.fc.in_features
            self.net.fc = nn.Linear(in_feats, 9)
            self.net.avgpool = nn.AdaptiveAvgPool2d(1)
        

        '''
        self.net = models.resnet18(pretrained=True)
        self.device = device
        in_feats = self.net.fc.in_features
        self.net.fc = nn.Linear(in_feats, 9)

        '''



        self.net.load_state_dict(torch.load(pose_cnn_param_file))
        #self.net.to(self.device)

        self.net = nn.DataParallel(self.net).cuda()
        self.net.eval()

    def predict(self, I):

        height = I.shape[0]
        width = I.shape[1]
        diff = int(round((height - width) / 2))

        if diff > 0:
            sq_image = hstack([128 * np.ones((height, diff, 3)), I, 128 * np.ones((height, diff, 3))])
        else:
            sq_image = vstack([128 * np.ones((-diff, width, 3)), I, 128 * np.ones((-diff, width, 3))])

        # resize image
        imsize = (227, 227)
        sq_image = cv2.resize(sq_image, imsize)

        '''
        image = sq_image
        if image.ndim == 3:
            image = np.expand_dims(image, 0)
        assert image.ndim == 4

        # If format is (N, H, W, 3) convert to (N, 3, H, W)
        # or raise an error if incorrect num of dimmensions.
        if image.shape[1] != 3:
            if image.shape[3] == 3:
                image = image.transpose((0, 3, 1, 2))
            else:
                raise RuntimeError("Wrong size of dimmensions. \ Correct size is (BatchSize, Height, With, 3)")

        images = (image.astype(np.float32) / 128) - 1
        imagesT = torch.tensor(images, dtype=torch.float32, device=self.device)
        '''

        #image = (image.astype(np.float32) / 128) - 1
        
        image = sq_image.transpose((2,0,1))
        image = image[np.newaxis,:,:,:]
        image = torch.from_numpy(image).float()
        image = (image/ 128) - 1
        
        image = Variable(image.cuda())

        #image = idc.cv2torch(sq_image)
        #image = image.type(torch.FloatTensor)
        #image = (image / 128) - 1


        outputs = self.net(image)
        R = outputs.data.cpu().numpy().reshape((3,3))  # axis angle
        u,s,vt = np.linalg.svd(R)
        
        R = np.dot(u,vt).transpose()
        #R_det = np.linalg.det(R)
        #D = np.diag([1,1,R_det])
        #R = np.dot(u,np.dot(D,vt)).transpose()
        

        #R = cv2.Rodrigues(aa)[0]  # convert from axis angle to rotation  (you could also use R= Axis2Rot(aa)
        return R



class Detector(object):

    def __init__(self, ssd_param_file, size=300, num_classes=2,  use_gpu = False):

        self.net = build_ssd('test', size, num_classes)  # initialize
        param = torch.load(ssd_param_file)
        self.net.load_state_dict(param)
        self.net.eval()
        self.size =size

        self.use_gpu = use_gpu
        if self.use_gpu:
            self.net = self.net.cuda()


    def detect(self, img):

        imsize = np.asarray((img.shape[0], img.shape[1]))
        img = cv2.resize(np.array(img), (self.size, self.size)).astype(np.float32)
        img -= 128
        img = img.astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img.unsqueeze(0)
        img = Variable(img)

        if self.use_gpu:

            img = img.cuda()

        else:
            pass

        x = self.net(img)

        # Todo: need non-maximum suppression? Currently, just selecting a box with max probability
        detections = x.data
        detections_ = detections.view(torch.Size([-1, 5]))
        d = detections_[:, 0]
        values, indices = torch.max(detections_[:, 0], 0)
        bbox = detections_[indices, [1, 2, 3, 4]].cpu().numpy()

        bbox[0] = np.round(bbox[0] * imsize[1])
        bbox[1] = np.round(bbox[1] * imsize[0])
        bbox[2] = np.round(bbox[2] * imsize[1])
        bbox[3] = np.round(bbox[3] * imsize[0])
        bbox = bbox.astype(int)

        bbox[0] = max(1, bbox[0])
        bbox[1] = max(1, bbox[1])
        bbox[2] = min(bbox[2], imsize[1])
        bbox[3] = min(bbox[3], imsize[0])

        return bbox



class ICP(object):

    def __init__(self, target, model, R_init, t_init):

        
        model.Transform(R_init, t_init)

        # shift back slightly to account for the distance between surface and centroid

        view = - model.centroid / np.linalg.norm(- model.centroid)
        angle = np.arccos(np.sum(view * model.v_normal, axis=1))*180 /pi
        front_facing_idx=nonzero(angle < 100)
        model_front_facing = model.vertex[front_facing_idx]
        #t = model.centroid - np.mean(model_front_facing, axis=0)
        #model.Transform(np.identity(3), t)

        self.target = target
        self.model = model
        self.kdtree = cKDTree(self.target.vertex)
        self.model_init = np.copy(model.vertex)  # copy the initial for the later rendering use
        self.R = R_init
        self.t = t_init

    def calculate(self, maxIter):

        for i in range(maxIter):

            view = - self.model.centroid / np.linalg.norm(- self.model.centroid)
            #print(view.shape)
            #print(self.model.v_normal.shape)

            angle = np.arccos(np.sum(view * self.model.v_normal, axis=1))*180 /pi
            front_facing_idx=nonzero(angle < 100)

            model_front_facing = self.model.vertex[front_facing_idx]

            #print(model_front_facing.shape[0])
            neighbor_idx = self.kdtree.query(model_front_facing)[1]
            closest_points = self.target.vertex[neighbor_idx]
            closest_point_normals = self.target.v_normal[neighbor_idx]
            vec = (self.model.vertex[front_facing_idx] - closest_points)
            dist = np.sqrt(np.sum(vec * vec, axis = 1))



            
            if i < maxIter/2:         # step1: translation only

                idxNear = nonzero(dist < 0.2* self.model.bbox)
                T = tf.calcTranslation(model_front_facing[idxNear], closest_points[idxNear])
                R = np.identity(3)

            elif i >  2* maxIter / 3: # step3: rotatation + translation
            #else:
                idxNear = nonzero(dist < 0.05 * self.model.bbox)
                #R, T = tf.calcRigidTranformation(model_front_facing[idxNear], closest_points[idxNear])
                #print(closest_point_normals[idxNear].shape)
                #print(closest_points[idxNear].shape)
                R, T = tf.calcRigidTranformationPointToPlane(model_front_facing[idxNear], closest_points[idxNear],  closest_point_normals[idxNear])

            else:
                idxNear = nonzero(dist < 0.1 * self.model.bbox)
                #R, T = tf.calcRigidTranformation(model_front_facing[idxNear], closest_points[idxNear])
                R, T = tf.calcRigidTranformationPointToPlane(model_front_facing[idxNear], closest_points[idxNear],  closest_point_normals[idxNear])

                #idxNear = nonzero(dist < 0.05 * self.model.bbox)
                #T = tf.calcTranslation(model_front_facing[idxNear], closest_points[idxNear])
                #R = np.identity(3)

            #else:                  # step2: rotatationnal  (without twist) + translation
            #    idxNear = nonzero(dist < 0.1 * self.model.bbox)
            #    R, T = tf.calcRigidTranformationPrincipalAxis(model_front_facing[idxNear], closest_points[idxNear],self.model.principalAxis)
            
            #idxNear = nonzero(dist < 0.02 * self.model.bbox)
            #R, T = tf.calcRigidTranformation(model_front_facing[idxNear], closest_points[idxNear])
            #idxNear = nonzero(dist < 0.3 * self.model.bbox)
            #T = tf.calcTranslation(model_front_facing[idxNear], closest_points[idxNear])
            #R = np.identity(3)


            self.model.Transform(R, T)
            self.R = np.dot(R, self.R)
            self.t = np.dot(R, np.reshape(self.t ,(3,1))) + np.reshape(T,(3,1))

        err = np.mean(dist)
        return err




if __name__ == '__main__':

    idxfile =  int(argv[1])
    use_gpu = torch.cuda.is_available()

    # +++++++++++++++++++++++++++++ file paths +++++++++++++++++++++++++++++
    ssd_param_file = "models/ssd_tool_yellow.pth"
    pose_cnn_param_file = "models/model.dat"

    #imageFile = "dataset/kinect_data/frame0000.jpg"

    imageFile = "dataset/tool20170123_rcnn/cloud%03d.png" % (idxfile)
    #pcFile = "dataset/kinect_data/frame0000.pcd"
    pcFile = "dataset/tool20170123_rcnn/cloud%03d.pcd" % (idxfile)

    #imageFile = "dataset/20180123/cropped%04d.jpg" % (idxfile)
    #pcFile = "dataset/20180123/cropped%04d.pcd" % (idxfile)

    cad_file = "dataset/dewalt_dcs551d2_with_dcs203_surf_no_col.off"
    principalAxis_yellow = np.asarray([0,0,1]) # principal axis = z axis, which is used in ICP

    print('reading...' + imageFile + "\n")
    print('reading...' + pcFile + "\n")

    v, f = readOff.read_off(cad_file)  # CAD
    V = np.asarray(v)
    F = np.asarray(f)
    cad_center = np.mean(V, 0)
    # +++++++++++++++++++++++++++++ file paths +++++++++++++++++++++++++++++


    # +++++++++++++++++++++++++++++ parameters +++++++++++++++++++++++++++++
    Rc = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # this rotation matrix aligns camera coords with depth coords
    #Rc = np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]])  # this rotation matrix aligns camera coords with depth coords

    icpMaxIter = 100
    ssd_size= 300
    num_classes= 2
    # +++++++++++++++++++++++++++++ parameters +++++++++++++++++++++++++++++


    # +++++++++++++++++++++++++++++ load nets +++++++++++++++++++++++++++++
    detector = Detector(ssd_param_file, ssd_size, num_classes, use_gpu)
    pose = PosePredictor(pose_cnn_param_file, use_gpu)
    img = cv2.imread(imageFile)
    # +++++++++++++++++++++++++++++ load nets +++++++++++++++++++++++++++++


    # +++++++++++++++++++++++++++++ ssd +++++++++++++++++++++++++++++
    start = time.time()
    bbox = detector.detect(img)


    img_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]



    elapsed_time = time.time() - start
    print("ssd forward time is: " + str(elapsed_time) + "sec" + "\n")
    # +++++++++++++++++++++++++++++ ssd +++++++++++++++++++++++++++++

    print(bbox)


    # +++++++++++++++++++++++++++++ pose cnn +++++++++++++++++++++++++++++
    start = time.time()
    R = pose.predict(img_crop)
    elapsed_time = time.time() - start
    print("poseCNN forward time is: " + str(elapsed_time) + "sec" + "\n")
    # +++++++++++++++++++++++++++++ pose cnn +++++++++++++++++++++++++++++


    # +++++++++++++++++++++++++++++ preprocess depth +++++++++++++++++++++++++++++
    pc_crop, pc_cylinder = PreprocessPointCloud(pcFile, bbox)
    # +++++++++++++++++++++++++++++ preprocess depth +++++++++++++++++++++++++++++


    # +++++++++++++++++++++++++++++ ICP +++++++++++++++++++++++++++++
    start = time.time()
    target = np.asarray(pc_cylinder)[:, 0:3]
    model_init = geo.Mesh(V - cad_center, F, principalAxis_yellow)
    R_init = np.dot(Rc, R)
    t_init = np.median(target, 0)   # using median to compute the centroid of point cloud as there are outliers but I'm not sure if this is okay



    icp = ICP(target, model_init, R_init, t_init)
    err = 0
    err = icp.calculate(icpMaxIter) # 100 steps
    elapsed_time = time.time() - start

    print("ICPtime is: " + str(elapsed_time) + "sec" +  "\n")
    print("ICP error is:" + str(err) + "\n")
    # +++++++++++++++++++++++++++++ ICP +++++++++++++++++++++++++++++


    print(icp.R)
    print(icp.t)

    VisualizeICPResult(imageFile, bbox, icp.model, icp.model_init, icp.target, pc_crop)

