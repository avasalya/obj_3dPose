#!/usr/bin/env python

# TODO
#Apply color ICP 
#visualization of point cloud and mesh 
#Input as parameter 1)dist_coeff 2)camera_matrix 3)name 4)make changes to launch file and remove unwanted params



#----------------------------Library--------------------------------------
import open3d.open3d as o3d
import torch as torch

import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from mesh_publisher import CMeshPublisher

import sys
# sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages') # in order to import cv2 under python3
# sys.path.append('/usr/lib/python3/dist-packages') # in order to import cv2 under python3
# sys.path.append('/home/ash/catkin_ws/devel/lib/python3/dist-packages') # in order to import cv2 under python3
# sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages') # append back in order to import rospy

# import pcl
import time

import copy
import numpy as np

from sensor_msgs.point_cloud2 import read_points
from std_msgs.msg import Int64

from lib_cloud_conversion import convertCloudFromOpen3dToRos 
from ppc import PreprocessPointCloud
from geometry_msgs.msg import PoseStamped
import tools.Geometry as geo
import tools.readOff as readOff
# from tf.transformations import quaternion_from_matrix # not used?

from darknet_ros_msgs.msg import BoundingBoxes

import cv2
from cv_bridge import CvBridge

from object_icp import VisualizeICPResult, PosePredictor, Detector, ICP


print('\n')
print('\n')
print('\n')
print('\n')
print('\n')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)

# +++++++++++++++++++++++++++++ parameters +++++++++++++++++++++++++++++

icpMaxIter = rospy.get_param('icpMaxIter', 100) 
namespace = rospy.get_namespace()
name = ["lipton_milk", "cupnoodle"]

cad_file = rospy.get_param(namespace + 'object_3d_pose/cad_file')
pose_cnn_param_file = rospy.get_param(namespace + 'object_3d_pose/pose_cnn_param_file')

#Kinect V2 params
# dist_coeff = np.asarray([0.07743601877351856, -0.12868103588600188, 0.0006922986358800922, 0.0006377742816840374, 0.051758649714233385])
# camera_matrix =  np.asarray([533.8407095878345, 0.0, 480.8396713281674, 0.0, 535.2956793798469, 282.6062774674073, 0.0, 0.0, 1.0]).reshape((3,3))

#Astra params
dist_coeff = np.asarray([0.03054091399141645, -0.1167877268666413, 0.007821888600736474, -0.008449467792505284, 0.0])
camera_matrix =  np.asarray([525.0544915325629, 0.0, 312.037380209421, 0.0, 524.497818208426, 243.5628499190073, 0.0, 0.0, 1.0]).reshape((3,3))


class object_3dpose(object):
   
    def __init__(self):

        self.last_image_msg = None
        self.last_cloud_msg = None
        self.pc_loc = None
        # self.height_pc = None
        # self.width_pc = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bridge = CvBridge()
        # self.bridge = cv_bridge_py36.CvBridge()

        # +++++++++++++++++++++++++++++ load nets ++++++++++++++++++++++
        print("Pose CNN parameter file", pose_cnn_param_file)
        self.pose_detector = PosePredictor(pose_cnn_param_file, self.device)

        # ++++++++++++++++++++++  set subscribers ++++++++++++++++++++++
        self.image_sub = rospy.Subscriber(rospy.get_param(namespace + "object_3d_pose/image_topic"), Image, self.image_cb, queue_size=1)
        self.cloud_id_sub = rospy.Subscriber("pcd_save_done", Int64, self.cloud_cb, queue_size = 1)
        # self.clou_sub = rospy.Subscriber(rospy.get_param(namespace + "object_3d_pose/cloud_topic"), PointCloud2, self.cloud_cb, queue_size=1)
        self.bbox_sub = rospy.Subscriber(rospy.get_param(namespace + "object_3d_pose/bbox_topic"), BoundingBoxes, self.bbox_cb, queue_size=1)

        # ++++++++++++++++++++++  set publishers ++++++++++++++++++++++
        self.pose3d_pub = rospy.Publisher(namespace + 'object_3d_pose/tool_pose', PoseStamped, queue_size=1)
        self.crop_pub = rospy.Publisher(namespace + 'object_3d_pose/crop_img', Image, queue_size=1)
        self.pcl_crop_pub = rospy.Publisher(namespace + 'object_3d_pose/debug_pcl', PointCloud2, queue_size=1)     
        self.model_pub = rospy.Publisher(namespace + 'object_3d_pose/model_pub', PointCloud2, queue_size=1)
        self.model_initial_pub = rospy.Publisher(namespace + 'object_3d_pose/model_initial_pub', PointCloud2, queue_size=1)       
       

        # ++++++++++++++++++++++  Mesh model for object in names ++++++++++++++++++++++

        self.mesh_model= o3d.io.read_triangle_mesh(str(cad_file)) # load
        self.pcd_model = o3d.io.read_point_cloud(str(cad_file))
        center_mesh  = self.mesh_model.get_center()
        center_pcd   = self.pcd_model.get_center()
        ratio=1.0/1000 # 1000 - the size in not matching for kinect v2 (900 working)
        transformation_mesh = np.asarray([[1*ratio, 0, 0, -center_mesh[0]],
                                        [ 0, 1*ratio, 0, -center_mesh[1]],
                                        [ 0, 0, 1*ratio, -center_mesh[2]],
                                        [ 0, 0, 0, 1]])

        transformation_pcd = np.asarray([[1*ratio, 0, 0, -center_pcd[0]],
                                        [ 0, 1*ratio, 0, -center_pcd[1]],
                                        [ 0, 0, 1*ratio, -center_pcd[2]],
                                        [ 0, 0, 0, 1]])

        self.mesh_model.transform(transformation_mesh)
        self.pcd_model.transform(transformation_pcd)
        print ([self.pcd_model])


        print("configured")
        return

    def image_cb(self, msg):

        self.last_image_msg = msg
    
    def cloud_cb(self,msg):
        cloud_loc = "/home/ash/catkin_ws/src/object_3dpose/data/" + str(msg.data) + ".pcd"
        # print ("reading_pc", msg.data)
        self.pc_loc = msg.data


        self.last_cloud_msg = o3d.io.read_point_cloud(cloud_loc,remove_nan_points=False, remove_infinite_points=False)
        # print (self.last_cloud_msg)
        # print (msg.height)
        # print (msg.width)

    def bbox_cb(self,msg):

        if self.last_cloud_msg is None or self.last_image_msg is None:
            print("no img or pcd rcvd")
            return

        img = self.bridge.imgmsg_to_cv2(self.last_image_msg, "bgr8" )#[:,:,::-1]  # ros msg to img
        # print ("msg",msg)
        pointcld = self.last_cloud_msg
        print ("reading_pc", self.pc_loc)



        for bb in msg.bounding_boxes:
            if bb.Class in name:
                # print("bb",[bb.xmin,bb.xmax,bb.ymin,bb.ymax])
                print (bb.Class)
                bbox_ = bb
                break 

        img_crop = img[bbox_.ymin:bbox_.ymax, bbox_.xmin:bbox_.xmax]
        # print img_crop.shape
        # cv2.imshow("Image Window", img_crop)
        # cv2.waitKey(1)


        self.crop_pub.publish(self.bridge.cv2_to_imgmsg(img_crop, "bgr8" ))

        bbox = [bbox_.xmin, bbox_.ymin, bbox_.xmax, bbox_.ymax]

        R_init = self.pose_cnn(img_crop)
        # print ("R_init", R_init)

        pc_crop, tvec = self.preprocess(bbox, R_init, pointcld)
        # print ("tvec", tvec)

        self.cicp_algorithm_reversed(pc_crop, tvec, R_init)
        # self.test_transformation(pc_crop, tvec, R_init)

    def pose_cnn(self, img_crop):

        start = time.time()
        R = self.pose_detector.predict(img_crop)
        elapsed_time = time.time() - start
        # print("poseCNN forward time is: " + str(elapsed_time) + "sec" + "\n")

        Rc = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # this rotation matrix aligns camera coords with depth coords
        # Rc = np.eye(3)#np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]])  # this rotation matrix aligns camera coords with depth coords
        #Rc = np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        #print(R)
        R_init = np.dot(Rc, R)
        # print ("R_init", R_init)

        return R_init

    def preprocess(self, bbox, R_init, pointcld):

        # print (self.mesh_model)
        V = np.asarray(self.mesh_model.vertices) # option of using vertex_color in o3d
        F = np.asarray(self.mesh_model.triangles)
        # print ("V", V.shape)
        # print ("F", F)

        bbox2 = bbox # expand the bb
        bbox2[0] = bbox2[0] - 10 
        bbox2[1] = bbox2[1] - 10
        bbox2[2] = bbox2[2] + 10
        bbox2[3] = bbox2[3] + 10
        # print ("bbox2", bbox2)

        bbox3 = bbox # shrink the bb
        bbox3[0] = bbox3[0] + 10 
        bbox3[1] = bbox3[1] + 10
        bbox3[2] = bbox3[2] - 10
        bbox3[3] = bbox3[3] - 10   

        #corner points for bb
        corners = np.asarray([ [bbox3[0], bbox3[1]], [bbox3[0], bbox3[3]], [bbox3[2], bbox3[3]], [bbox3[2], bbox3[1]] ], np.float32)
        # print ("corners", corners)

        #corner points for cad mesh
        V_rot = np.dot(R_init, V.transpose()).transpose() #rotation of self.mesh_model
        # print ("V_rot", V_rot)
        bbox3D = np.asarray([np.min(V_rot,axis =0), np.max(V_rot, axis =0)]) #minimum and maximum point 
        # print ("bbox3D", bbox3D)
        # bbox3D = [bbox3D[0][0],bbox3D[0][1]-0.02,bbox3D[1][0],bbox3D[1][1]] #0.02 ?
        bbox3D = [bbox3D[0][0],bbox3D[0][1],bbox3D[1][0],bbox3D[1][1]] #xmin, ymin, xmax, ymax
        corners3D = np.asarray([[bbox3D[0], bbox3D[1], 0], [bbox3D[0], bbox3D[3], 0], [bbox3D[2], bbox3D[3], 0], [bbox3D[2], bbox3D[1], 0]] ) # bb plane at z=0
        # print ("corners3D", corners3D)

        est = cv2.solvePnP(corners3D, corners, camera_matrix, dist_coeff)
        rvec =est[1]
        tvec =est[2]
        # print("rvec", rvec)
        # print("tvec", tvec)

        height = rospy.get_param(namespace + "/object_3d_pose/point_cloud_height")
        width = rospy.get_param(namespace + "/object_3d_pose/point_cloud_width")


        depth_range = 0.5;
        limits = [tvec[0]-depth_range, tvec[0]+depth_range,tvec[1]-depth_range,tvec[1]+depth_range,tvec[2]-depth_range,tvec[2]+depth_range]
        pc_crop = PreprocessPointCloud(pointcld, bbox2, height, width, limits)
        # o3d.io.write_point_cloud("/home/jrluser/pc_crop_bb.pcd", pc_crop)

        return pc_crop, tvec






        """ problem is with  below three lines during publishing back point cloud data to ROS
                self.pcl_crop_pub.publish(target_temp_ros)
                    self.model_pub.publish(source_temp_ros)
                else:
                self.model_initial_pub.publish(source_temp_ros)
        """ 

        """ this method is from the git repo for ROS to open3d pointcloud 

            convertCloudFromOpen3dToRos

        """

    def draw_registration(self,source, target, transformation):

        sensor_frame = "camera_rgb_optical_frame" #Astra
        # sensor_frame = "kinect2_rgb_optical_frame" #Kinect
        # print ([target])
        source_temp = copy.deepcopy(source) # model copy
        source_temp.transform(transformation)
        source_temp_ros = convertCloudFromOpen3dToRos(source_temp,sensor_frame)

        if target != None:
            target_temp = target #copy.deepcopy(target) -- check if required to copy target point cloud
            #ROS
            target_temp_ros = convertCloudFromOpen3dToRos(target_temp,sensor_frame)
            self.pcl_crop_pub.publish(target_temp_ros)
            self.model_pub.publish(source_temp_ros)

        else:
            self.model_initial_pub.publish(source_temp_ros)

        # o3d.io.write_point_cloud("/home/jrluser/Desktop/project/test_cicp_tuning_pipeline/pc_model.pcd",source_temp)
        # o3d.io.write_point_cloud("/home/jrluser/Desktop/project/test_cicp_tuning_pipeline/pc_milk_blue.pcd",target_temp)


        #Open3d
        # source_temp.paint_uniform_color([0, 0, 1])
        # target_temp.paint_uniform_color([0, 1, 0])
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1)
        # o3d.visualization.draw_geometries([source_temp, target_temp])#, mesh_frame])




    def icp_algorithm(self,pc_crop, tvec, R_init):

        pcd_model_ds = self.pcd_model.voxel_down_sample(voxel_size=0.001) #reduce number of points
        pc_crop = pc_crop.voxel_down_sample(voxel_size=0.001)
        print ("Point to plane ICP")
        threshold = 0.02
        radius = 0.05
        camera = [0, 0, 0]
        current_transformation = np.r_[np.c_[R_init,tvec], np.array([[0.0,0.0,0.0,1.0]])]
        source_temp = copy.deepcopy(pcd_model_ds) # model copy
        source_temp.transform(current_transformation)

        hull, list_1 = source_temp.hidden_point_removal(camera,10000) #cut the model to remove back side
        source_temp_front = source_temp.select_down_sample(list_1)

        current_transformation = np.identity(4)

        # self.draw_registration(target_temp_front, None, current_transformation)
        
        pc_crop.estimate_normals( o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))
        source_temp_front.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))
        
        result_icp = o3d.registration.registration_icp(
                            source_temp_front, pc_crop, threshold, current_transformation,
                            o3d.registration.TransformationEstimationPointToPlane(),
                            o3d.registration.ICPConvergenceCriteria(max_iteration=200))

        print(result_icp.fitness)
        # if (result_icp.fitness != 0.0): 
        #     print ("Updating data")
        self.draw_registration(source_temp_front, pc_crop,
                                     result_icp.transformation)

    def icp_algorithm_reversed(self,pc_crop, tvec, R_init):
        pcd_model_ds = self.pcd_model #.voxel_down_sample(voxel_size=0.0015)
        camera = [0, 0, 0]
        # pc_crop = pc_crop.voxel_down_sample(voxel_size=0.0015)
        print (pc_crop)
        print ("Point to plane ICP reversed started ")
        threshold = 0.05 #0.02
        radius = 0.05 #0.05
        # print np.dot(-1*R_init.transpose(), tvec)
        current_transformation = np.r_[np.c_[R_init,tvec], np.array([[0.0,0.0,0.0,1.0]])]
        target_temp = copy.deepcopy(pcd_model_ds) # model copy
        target_temp.transform(current_transformation)

        hull, list_1 = target_temp.hidden_point_removal(camera,10000) #cut the model to remove back side
        target_temp_front = target_temp.select_down_sample(list_1)

        # current_transformation = np.r_[np.c_[R_init.transpose(),-1*np.dot(R_init.transpose(), tvec)], np.array([[0.0,0.0,0.0,1.0]])]
        # self.draw_registration(pc_crop, None, current_transformation)
        current_transformation = np.identity(4)
        pc_crop.estimate_normals( o3d.geometry.KDTreeSearchParamHybrid(radius= 2*radius, max_nn=50))
        target_temp_front.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))
        
        result_icp = o3d.registration.registration_icp(
                            pc_crop, # source
                            target_temp_front, #target
                            threshold,
                            current_transformation,
                            o3d.registration.TransformationEstimationPointToPlane(),
                            o3d.registration.ICPConvergenceCriteria(max_iteration=200))

        print(result_icp)
        # if (result_icp.fitness > 0.3): 
        #     print ("Updating data")

        rotation = result_icp.transformation[0:3, 0:3]
        translation = result_icp.transformation[0:3,3]
        # print rotation
        # print translation
        final_transformation = np.r_[np.c_[rotation.transpose(),-1*np.dot(rotation.transpose(), translation)], np.array([[0.0,0.0,0.0,1.0]])]
        self.draw_registration(target_temp_front, pc_crop,
                                         final_transformation)

    def test_transformation(self,pc_crop, tvec, R_init):

        current_transformation = np.r_[np.c_[R_init,tvec], np.array([[0.0,0.0,0.0,1.0]])]
        self.draw_registration(self.pcd_model, pc_crop,current_transformation)

    def cicp_algorithm(self,pc_crop, tvec, R_init):
        pcd_model_ds = self.pcd_model#.voxel_down_sample(voxel_size=0.002) #reduce number of points
        print ("#########################################################################")
        camera = [0, 0, 0]

        # voxel_radius = [0.005,0.0025, 0.002]
        # max_iter = [50, 30, 14]
        # geometric_param = [1.0, 0.968, 0.968, 0.968]
        voxel_radius = [0.004,0.0022, 0.0017]
        max_iter = [100, 50, 14]
        geometric_param = [0.968, 0.968, 0.968]

        # current_transformation =result_icp.transformation
        current_transformation = np.r_[np.c_[R_init,tvec], np.array([[0.0,0.0,0.0,1.0]])]
        # self.draw_registration(pcd_model_ds, None, current_transformation)

        source_temp = copy.deepcopy(pcd_model_ds) # model copy
        source_temp.transform(current_transformation)

        current_transformation = np.identity(4)

        hull, list_1 = source_temp.hidden_point_removal(camera,10000) #cut the model to remove back side
        source_temp_front = source_temp.select_down_sample(list_1)

        print ("")
        print("Colored point cloud registration")
        for scale in range(2):
            iter = max_iter[scale]
            radius = voxel_radius[scale]
            print([iter, radius, scale])

            print("1. Downsample with a voxel size ", radius)
            source_down = source_temp_front.voxel_down_sample(radius)
            target_down = pc_crop.voxel_down_sample(radius)

            print("2. Estimate normal.")
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=50))
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 10, max_nn=50))

            print("3. Applying colored point cloud registration")
            result_icp = o3d.registration.registration_colored_icp(
                source_down, target_down, 10*radius, current_transformation,
                o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                        relative_rmse=1e-6,
                                                        max_iteration=iter), lambda_geometric=geometric_param[scale])
            current_transformation = result_icp.transformation
            print(result_icp)

        self.draw_registration(source_temp_front, pc_crop,
                                         result_icp.transformation)

    def cicp_algorithm_reversed(self,pc_crop, tvec, R_init):
        # model in cut in front view 
        # pcd_model_ds = self.pcd_model#.voxel_down_sample(voxel_size=0.002) #reduce number of points
        print ("#########################################################################")
        print (pc_crop)
        camera = [0, 0, 0]
        voxel_radius = [0.003,0.0025, 0.0017]
        max_iter = [200, 50, 14]
        geometric_param = [1.0, 0.968, 0.968]
        current_transformation = np.r_[np.c_[R_init,tvec], np.array([[0.0,0.0,0.0,1.0]])]
        
        target_temp = copy.deepcopy(self.pcd_model) # model copy
        target_temp.transform(current_transformation)

        hull, list_1 = target_temp.hidden_point_removal(camera,10000)
        target_temp_front = target_temp.select_down_sample(list_1)

        # current_transformation =result_icp.transformation
        current_transformation =np.identity(4)
        # current_transformation = np.r_[np.c_[R_init.transpose(),-1*np.dot(R_init.transpose(), tvec)], np.array([[0.0,0.0,0.0,1.0]])]
        self.draw_registration(target_temp, None, current_transformation)
        # print ("")
        # testVar = raw_input("input to continue")
        print("Colored point cloud registration reversed1")
        for scale in range(3):
            iter = max_iter[scale]
            radius = voxel_radius[scale]
            print([iter, radius, scale])

            # print("1. Downsample with a voxel size ", radius)
            source_down = pc_crop.voxel_down_sample(radius)
            print (source_down)
            target_down = target_temp_front.voxel_down_sample(radius)
            print (target_down)


            # print("2. Estimate normal.")
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius*10, max_nn=50))
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2, max_nn=50))

            # print("3. Applying colored point cloud registration")
            result_icp = o3d.registration.registration_colored_icp(
                source_down, target_down, 20*radius, current_transformation,
                o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                        relative_rmse=1e-6,
                                                        max_iteration=iter), lambda_geometric=geometric_param[scale])
            current_transformation = result_icp.transformation
            print(result_icp)

        # if (result_icp.fitness != 0): 
        # print ("Updating data")
        # print (result_icp.transformation)
        rotation = result_icp.transformation[0:3, 0:3]
        translation = result_icp.transformation[0:3,3]
        # print rotation
        # print translation
        final_transformation = np.r_[np.c_[rotation.transpose(),-1*np.dot(rotation.transpose(), translation)], np.array([[0.0,0.0,0.0,1.0]])]
        # print final_transformation
        print ("$$$$$ drawing registration no.", scale + 1)
        self.draw_registration(target_temp, pc_crop, final_transformation)
        # testVar = raw_input("input to continue")




if __name__ == "__main__":
    rospy.init_node('3dpose_detector')
    detector = object_3dpose()
    rospy.spin()


'''
 def cicp_algorithm_reversed(self,pc_crop, tvec, R_init):

        pcd_model_ds = self.pcd_model#.voxel_down_sample(voxel_size=0.002) #reduce number of points
        print ("#########################################################################")
        print (pc_crop)

        voxel_radius = [0.0035,0.0022, 0.0017]
        max_iter = [100, 50, 14]
        geometric_param = [1.0, 0.968, 0.968]
        current_transformation = np.r_[np.c_[R_init,tvec], np.array([[0.0,0.0,0.0,1.0]])]
        target_temp = copy.deepcopy(pcd_model_ds) # model copy
        target_temp.transform(current_transformation)

        # current_transformation =result_icp.transformation
        current_transformation =np.identity(4)
        # current_transformation = np.r_[np.c_[R_init.transpose(),-1*np.dot(R_init.transpose(), tvec)], np.array([[0.0,0.0,0.0,1.0]])]
        # self.draw_registration(pc_crop, None, current_transformation)
        # print ("")
        print("Colored point cloud registration reversed")
        for scale in range(1):
            iter = max_iter[scale]
            radius = voxel_radius[scale]
            print([iter, radius, scale])

            # print("1. Downsample with a voxel size ", radius)
            source_down = pc_crop.voxel_down_sample(radius)
            print source_down
            target_down = target_temp.voxel_down_sample(radius)
            print target_down


            # print("2. Estimate normal.")
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius*10, max_nn=50))
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2, max_nn=50))

            # print("3. Applying colored point cloud registration")
            result_icp = o3d.registration.registration_colored_icp(
                source_down, target_down, 10*radius, current_transformation,
                o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                        relative_rmse=1e-6,
                                                        max_iteration=iter), lambda_geometric=geometric_param[scale])
            current_transformation = result_icp.transformation
            print(result_icp)

        if (result_icp.fitness != 0): 
            print ("Updating data")
            # print (result_icp.transformation)
            rotation = result_icp.transformation[0:3, 0:3]
            translation = result_icp.transformation[0:3,3]
            # print rotation
            # print translation
            final_transformation = np.r_[np.c_[rotation.transpose(),-1*np.dot(rotation.transpose(), translation)], np.array([[0.0,0.0,0.0,1.0]])]
            # print final_transformation
            self.draw_registration(target_temp, pc_crop, final_transformation)
'''
