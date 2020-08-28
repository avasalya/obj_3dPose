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
pose_cnn_param_file = rospy.get_param('./object3d_pose/pose_cnn_param_file')
cad_file = rospy.get_param('cad_file') 

dist_coeff = np.asarray([0.03071415200703961, -0.09421218949123161, 0.003948828921829864, 0.002806380632054664, 0.0])
camera_matrix =  np.asarray([521.8907476940199, 0.0, 331.6423101680497, 0.0, 523.9112560649733, 233.7618907087222, 0.0, 0.0, 1.0]).reshape((3,3))

# +++++++++++++++++++++++++++++ parameters +++++++++++++++++++++++++++++


class object_3dpose(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ++++++++++++++  set publishers and subscribers +++++++++++++++
        
        # +++++++++++++++++++++++++++++ load nets +++++++++++++++++++++++++++++
        self.pose_detector = PosePredictor(pose_cnn_param_file, self.device)
        # +++++++++++++++++++++++++++++ load nets +++++++++++++++++++++++++++++


        self.bridge = CvBridge()
        self.last_image_msg = None
        self.last_cloud_msg = None
        self.imag_sub = rospy.Subscriber(rospy.get_param("image_topic"), Image, self.image_cb, queue_size=1)
        self.clou_sub = rospy.Subscriber(rospy.get_param("cloud_topic"), PointCloud2, self.cloud_cb, queue_size=1)
        self.bbox_sub = rospy.Subscriber(rospy.get_param("bbox_topic"), BoundingBoxes, self.bbox_cb, queue_size=1)
        self.pose3d_pub = rospy.Publisher('tool_pose', PoseStamped, queue_size=1)
        self.crop_pub = rospy.Publisher('crop_img', Image, queue_size=1)
        self.pcl_crop_pub = rospy.Publisher('debug_pcl', PointCloud2, queue_size=1)        
        self.print_mesh = CMeshPublisher(rospy.get_param("mesh_file"),"mesh","camera_rgb_optical_frame")
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
        cloud_points = read_points(self.last_cloud_msg, skip_nans=False, field_names=("x", "y", "z"))
        pc_list = []
        for p in cloud_points:
            pc_list.append( [p[0],p[1],p[2]] )

        pc = pcl.PointCloud()
        pc.from_list(pc_list)



        img_crop = img[msg.bounding_boxes[0].ymin:msg.bounding_boxes[0].ymax, msg.bounding_boxes[0].xmin:msg.bounding_boxes[0].xmax] #[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        #cv2.imwrite('/home/toni/Pictures/test.jpg',img_crop)     
        #img_crop = cv2.imread('/media/toni/SONY_32GT/tool_yellow_pytorch/train/ar2578.jpg')[:,:,::-1] 
        #img_crop = cv2.imread('/home/toni/Pictures/test.jpg')[:,:,::-1] 

        self.crop_pub.publish(self.bridge.cv2_to_imgmsg(img_crop))
        bbox = [msg.bounding_boxes[0].xmin,msg.bounding_boxes[0].ymin,msg.bounding_boxes[0].xmax,msg.bounding_boxes[0].ymax]

        bbox2 = bbox
        bbox2[0] = bbox2[0] - 5 
        bbox2[1] = bbox2[1] - 5
        bbox2[2] = bbox2[2] + 5
        bbox2[3] = bbox2[3] + 5
        # +++++++++++++++++++++++++++++ pose cnn +++++++++++++++++++++++++++++
        start = time.time()
        R = self.pose_detector.predict(img_crop)
        elapsed_time = time.time() - start
        print("poseCNN forward time is: " + str(elapsed_time) + "sec" + "\n")
        # +++++++++++++++++++++++++++++ pose cnn +++++++++++++++++++++++++++++



        # +++++++++++++++++++++++++++++ ICP +++++++++++++++++++++++++++++
        Rc = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # this rotation matrix aligns camera coords with depth coords
        #Rc = np.eye(3)#np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]])  # this rotation matrix aligns camera coords with depth coords
        
        #Rc = np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        start = time.time()


        principalAxis_yellow = np.asarray([0,0,1]) # principal axis = z axis, which is used in ICP
        v, f = readOff.read_off(cad_file)  # CAD
        V = np.asarray(v)
        F = np.asarray(f)
        cad_center = np.mean(V, 0)
        #model_init = geo.Mesh(V , F, principalAxis_yellow)
        model_init = geo.Mesh(V, F, principalAxis_yellow)

        #R = np.asarray([-0.342020143, 0.89370079,  0.290380989, -0.939692621,    -0.325280486,    -0.105690037 ,   2.78E-17,    -0.309016994 ,   0.951056516]).reshape((3,3)).transpose() #8.jpg
        #R = np.asarray([0.939692621, 0.241844763, 0.241844763, -0.342020143, 0.664463024, 0.664463024, 1.39E-17, -0.707106781, 0.707106781]).reshape((3,3)).transpose() #53.jpg
        #R = np.asarray([-0.004125865,-0.650128507,-0.75981307, -0.134487097, -0.752555974,0.644649305, -0.990906755, 0.10484479,  -0.084328952]).reshape((3,3)).transpose() 
        #R = np.asarray([-0.185202,   -0.874768,   -0.447751,   -0.00197658, -0.455301,   0.890335,    -0.982698,   0.165777,    0.0825937]).reshape((3,3)).transpose() 
        #R = np.asarray([-0.931585,   -0.329211 ,  -0.154172 ,  0.00915838 , -0.445225  , 0.895372  ,  -0.363408  , 0.832703   , 0.41778]).reshape((3,3)).transpose() 
        #R = np.asarray([0.933773,    0.325102 ,   0.149587,    0.0121103 ,  -0.446465 ,  0.894719  ,  0.357661 ,   -0.833653 ,  -0.420834]).reshape((3,3)).transpose()
        #R = np.asarray([0.874868 , -0.426461 ,-0.229648 , -0.0138401, 0.45192, -0.891951, 0.484164, 0.783517, 0.389468]).reshape((3,3)).transpose()


        print(R)
        R_init = np.dot(Rc, R)
        #t_init = np.median(target, 0) + cad_center 
        #t_init[2] = t_init[2] + 0.05

        bbox3= bbox
        bbox3[0] = bbox3[0] + 10 
        bbox3[1] = bbox3[1] + 10
        bbox3[2] = bbox3[2] - 10
        bbox3[3] = bbox3[3] - 10   
        
        corners = np.asarray([ [bbox3[0], bbox3[1]], [bbox3[0], bbox3[3]], [bbox3[2], bbox3[3]], [bbox3[2], bbox3[1]] ], np.float32)
        V_rot = np.dot(R_init, V.transpose()).transpose()
        bbox3D = np.asarray([np.min(V_rot,axis =0), np.max(V_rot, axis =0)])
        bbox3D = [bbox3D[0][0],bbox3D[0][1]-0.02,bbox3D[1][0],bbox3D[1][1]]
        
        corners3D = np.asarray([[bbox3D[0], bbox3D[1], 0], [bbox3D[0], bbox3D[3], 0], [bbox3D[2], bbox3D[3], 0], [bbox3D[2], bbox3D[1], 0]] )

        #print(corners3D)
        #print(corners.shape[1])

        est = cv2.solvePnP(corners3D, corners, camera_matrix, dist_coeff)
        rvec =est[1]
        tvec =est[2]
        #print(rvec)
        #print(t_init)

        depth_range = 0.5;
        limits = [tvec[0]-depth_range, tvec[0]+depth_range,tvec[1]-depth_range,tvec[1]+depth_range,tvec[2]-depth_range,tvec[2]+depth_range]

                # +++++++++++++++++++++++++++++ preprocess depth +++++++++++++++++++++++++++++
        height = rospy.get_param("point_cloud_height")
        width = rospy.get_param("point_cloud_width")

        pc_crop, pc_cylinder = PreprocessPointCloud(pc, bbox2, height, width, limits)
        pc_cyl_array = np.asarray(pc_crop)[:, 0:3]
        pclmsg = PointCloud2()
        pclmsg.header.stamp = rospy.Time.now()
        pclmsg.header.frame_id = "camera_rgb_optical_frame"
        pclmsg.height = 1
        pclmsg.width = len(pc_cyl_array)
        pclmsg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        pclmsg.is_bigendian = False
        pclmsg.point_step = 12
        pclmsg.row_step = 12*pc_cyl_array.shape[0]
        pclmsg.is_dense = int(np.isfinite(pc_cyl_array).all())
        pclmsg.data = np.asarray(pc_cyl_array, np.float32).tostring()
        self.pcl_crop_pub.publish(pclmsg)



        pcd = o3d.PointCloud()
        pcd.points = o3d.Vector3dVector(np.asarray(pc_crop)[:, 0:3])
        o3d.estimate_normals(pcd, search_param = o3d.KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
        # +++++++++++++++++++++++++++++ preprocess depth +++++++++++++++++++++++++++++

        print(np.asarray(pcd.normals))

        t_init = np.asarray([tvec[0], tvec[1], tvec[2]]).reshape((1,3))

        #r = R_init
        
        # using median to compute the centroid of point cloud as there are outliers but I'm not sure if this is okay
        target = geo.Points(np.asarray(pcd.points), np.asarray(pcd.normals))

        
        icpMaxIter= 30
        icp = ICP(target, model_init, R_init, t_init)
        err = icp.calculate(icpMaxIter) # 100 steps
        
        err=0
        
        elapsed_time = time.time() - start

        #print(icp.t)
        #print(icp.R)


        print("ICPtime is: " + str(elapsed_time) + "sec" +  "\n")
        print("ICP error is:" + str(err) + "\n")
        # +++++++++++++++++++++++++++++ ICP +++++++++++++++++++++++++++++

        # ++++++++++++++++++++++++ return message ++++++++++++++++++++++++
        outmsg = PoseStamped()
        outmsg.header.frame_id = self.last_cloud_msg.header.frame_id
        outmsg.header.stamp = rospy.Time.now()
        outmsg.pose.position.x = icp.t[0]#tvec[0]
        outmsg.pose.position.y = icp.t[1]#tvec[1]
        outmsg.pose.position.z = icp.t[2]#tvec[2]
        fmatrix = np.eye(4)
        fmatrix[:3,:3] = icp.R
        orient_q = quaternion_from_matrix(fmatrix)
        outmsg.pose.orientation.x = orient_q[0]
        outmsg.pose.orientation.y = orient_q[1]
        outmsg.pose.orientation.z = orient_q[2]
        outmsg.pose.orientation.w = orient_q[3]

        self.print_mesh(outmsg.pose)
        self.pose3d_pub.publish(outmsg)   
        # ++++++++++++++++++++++++ return message ++++++++++++++++++++++++

if __name__ == "__main__":
    rospy.init_node('3dpose_detector')
    detector = object_3dpose()
    rospy.spin()
