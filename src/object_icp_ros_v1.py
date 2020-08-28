#!/usr/bin/env python

import torch
import rospy
import pcl
import time
import numpy as np
from object_icp import PreprocessPointCloud, VisualizeICPResult, PosePredictor, Detector, ICP
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
import tools.Geometry as geo
import tools.readOff as readOff
from mesh_publisher import CMeshPublisher
from tf.transformations import quaternion_from_matrix
from darknet_ros_msgs.msg import BoundingBoxes
from cv_bridge import CvBridge
from sensor_msgs.point_cloud2 import read_points

import open3d as o3d
import cv2
# +++++++++++++++++++++++++++++ parameters +++++++++++++++++++++++++++++

icpMaxIter = rospy.get_param('icpMaxIter', 100)

namespace = rospy.get_namespace()

cad_file = rospy.get_param(namespace + 'object_3d_pose/cad_file') 
pose_cnn_param_file = rospy.get_param(namespace + 'object_3d_pose/pose_cnn_param_file')

dist_coeff = np.asarray([0.03071415200703961, -0.09421218949123161, 0.003948828921829864, 0.002806380632054664, 0.0])
camera_matrix =  np.asarray([521.8907476940199, 0.0, 331.6423101680497, 0.0, 523.9112560649733, 233.7618907087222, 0.0, 0.0, 1.0]).reshape((3,3))

# +++++++++++++++++++++++++++++ parameters +++++++++++++++++++++++++++++


class object_3dpose(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ++++++++++++++  set publishers and subscribers +++++++++++++++
        #pose_cnn_param_file = rospy.get_param('/object_3d_pose/pose_cnn_param_file')

        print(pose_cnn_param_file)
        # +++++++++++++++++++++++++++++ load nets +++++++++++++++++++++++++++++
        self.pose_detector = PosePredictor(pose_cnn_param_file, self.device)
        # +++++++++++++++++++++++++++++ load nets +++++++++++++++++++++++++++++


        self.bridge = CvBridge()
        self.last_image_msg = None
        self.last_cloud_msg = None
        self.imag_sub = rospy.Subscriber(rospy.get_param(namespace + "object_3d_pose/image_topic"), Image, self.image_cb, queue_size=1)
        self.clou_sub = rospy.Subscriber(rospy.get_param(namespace + "object_3d_pose/cloud_topic"), PointCloud2, self.cloud_cb, queue_size=1)
        self.bbox_sub = rospy.Subscriber(rospy.get_param(namespace + "object_3d_pose/bbox_topic"), BoundingBoxes, self.bbox_cb, queue_size=1)
        self.pose3d_pub = rospy.Publisher(namespace + 'object_3d_pose/tool_pose', PoseStamped, queue_size=1)
        self.crop_pub = rospy.Publisher(namespace + 'object_3d_pose/crop_img', Image, queue_size=1)
        self.pcl_crop_pub = rospy.Publisher(namespace + 'object_3d_pose/debug_pcl', PointCloud2, queue_size=1)        
        self.print_mesh = CMeshPublisher(rospy.get_param(namespace + "object_3d_pose/mesh_file"),"mesh","camera_rgb_optical_frame")

        v, f = readOff.read_off(cad_file)  # CAD
        V = np.asarray(v)
        F = np.asarray(f)

        self.V = V
        self.F = F

        self.mesh = o3d.TriangleMesh()
        self.mesh.vertices = o3d.Vector3dVector(self.V)
        self.mesh.triangles = o3d.Vector3iVector(self.F)
        self.mesh.compute_vertex_normals()


        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()  
        self.vis.add_geometry(self.mesh)
        self.vis.run()



        return

    def image_cb(self, msg):
        self.last_image_msg = msg
    
    def cloud_cb(self,msg):
        self.last_cloud_msg = msg

    def bbox_cb(self,msg):
        if self.last_cloud_msg is None or self.last_image_msg is None:
            print("no image rcvd")
            return
        img = self.bridge.imgmsg_to_cv2(self.last_image_msg)#[:,:,::-1]  # ros msg to img
        
        name = 'lipton_lemon'#namespace[1:-1]
        for bb in msg.bounding_boxes:
            if bb.Class == name:
                print([bb.xmin,bb.xmax,bb.ymin,bb.ymax])
                bbox_ = bb
                break 

        img_crop = img[bbox_.ymin:bbox_.ymax, bbox_.xmin:bbox_.xmax] #[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        self.crop_pub.publish(self.bridge.cv2_to_imgmsg(img_crop))
        bbox = [bbox_.xmin, bbox_.ymin, bbox_.xmax, bbox_.ymax]

        bbox2 = bbox
        bbox2[0] = bbox2[0] - 10 
        bbox2[1] = bbox2[1] - 10
        bbox2[2] = bbox2[2] + 10
        bbox2[3] = bbox2[3] + 10
        # +++++++++++++++++++++++++++++ pose cnn +++++++++++++++++++++++++++++
        start = time.time()
        R = self.pose_detector.predict(img_crop)
        elapsed_time = time.time() - start
        #print("poseCNN forward time is: " + str(elapsed_time) + "sec" + "\n")
        # +++++++++++++++++++++++++++++ pose cnn +++++++++++++++++++++++++++++



        # +++++++++++++++++++++++++++++ ICP +++++++++++++++++++++++++++++
        Rc = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # this rotation matrix aligns camera coords with depth coords
        #Rc = np.eye(3)#np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]])  # this rotation matrix aligns camera coords with depth coords
        
        #Rc = np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        start = time.time()


        principalAxis_yellow = np.asarray([0,0,1]) # principal axis = z axis, which is used in ICP
        

        #R = np.asarray([-0.342020143, 0.89370079,  0.290380989, -0.939692621,    -0.325280486,    -0.105690037 ,   2.78E-17,    -0.309016994 ,   0.951056516]).reshape((3,3)).transpose() #8.jpg
        #R = np.asarray([0.939692621, 0.241844763, 0.241844763, -0.342020143, 0.664463024, 0.664463024, 1.39E-17, -0.707106781, 0.707106781]).reshape((3,3)).transpose() #53.jpg
        #R = np.asarray([-0.004125865,-0.650128507,-0.75981307, -0.134487097, -0.752555974,0.644649305, -0.990906755, 0.10484479,  -0.084328952]).reshape((3,3)).transpose() 
        #R = np.asarray([-0.185202,   -0.874768,   -0.447751,   -0.00197658, -0.455301,   0.890335,    -0.982698,   0.165777,    0.0825937]).reshape((3,3)).transpose() 
        #R = np.asarray([-0.931585,   -0.329211 ,  -0.154172 ,  0.00915838 , -0.445225  , 0.895372  ,  -0.363408  , 0.832703   , 0.41778]).reshape((3,3)).transpose() 
        #R = np.asarray([0.933773,    0.325102 ,   0.149587,    0.0121103 ,  -0.446465 ,  0.894719  ,  0.357661 ,   -0.833653 ,  -0.420834]).reshape((3,3)).transpose()
        #R = np.asarray([0.874868 , -0.426461 ,-0.229648 , -0.0138401, 0.45192, -0.891951, 0.484164, 0.783517, 0.389468]).reshape((3,3)).transpose()


        #print(R)
        R_init = R
        x = np.dot(R_init, self.V.transpose()).transpose()
 
        #mesh = o3d.TriangleMesh()
        self.mesh.vertices = o3d.Vector3dVector(x)
        self.mesh.triangles = o3d.Vector3iVector(self.F)
        self.mesh.compute_vertex_normals()

        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()


        #self.vis.add_geometry(mesh)
        self.vis.run()
        self.vis.destroy_window()

        #self.vis.remove_geometry(mesh)


        #t_init = np.median(target, 0) + cad_center 
        #t_init[2] = t_init[2] + 0.05


        #print(icp.t)
        #print(icp.R)


        #print("ICPtime is: " + str(elapsed_time) + "sec" +  "\n")
        #print("ICP error is:" + str(err) + "\n")
        # +++++++++++++++++++++++++++++ ICP +++++++++++++++++++++++++++++

        # ++++++++++++++++++++++++ return message ++++++++++++++++++++++++

        # ++++++++++++++++++++++++ return message ++++++++++++++++++++++++

if __name__ == "__main__":
    rospy.init_node('3dpose_detector')
    detector = object_3dpose()
    rospy.spin()
