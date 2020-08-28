#!/usr/bin/env python

import rospy
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Pose, Vector3
from visualization_msgs.msg import Marker

class CMeshPublisher(object):
    def __init__(self, meshfilename, topic, base_frame):
        self.mesh_marker = Marker()
        self.mesh_marker.header.frame_id = base_frame
        self.mesh_marker.ns = "Mesh" # unique ID
        self.mesh_marker.action = Marker().ADD
        self.mesh_marker.type = Marker().MESH_RESOURCE
        self.mesh_marker.mesh_resource = meshfilename
        self.mesh_marker.mesh_use_embedded_materials = True
        self.mesh_marker.id = 0
        self.pub_rviz_marker = rospy.Publisher(topic, Marker, queue_size=10)

    def __call__(self, pose, scale=1.0, color=None, duration=None):
    	#self.mesh_marker.id += 1
        self.mesh_marker.pose = pose
        # Convert input scale to a ROS Vector3 Msg
        if type(scale) == Vector3:
            self.mesh_marker.scale = scale
        elif type(scale) == float:
            self.mesh_marker.scale = Vector3(scale, scale, scale)
        else:
            rospy.logerr("Scale is unsupported type '%s' in publishMesh()", type(scale).__name__)
            return False
        if color == None:
            self.mesh_marker.color = ColorRGBA() # no color
        else:

            self.mesh_marker.mesh_use_embedded_materials = True
            self.mesh_marker.color.r = color[0]
            self.mesh_marker.color.g = color[1]
            self.mesh_marker.color.b = color[2]
            self.mesh_marker.color.a = color[3]
            print(color) # ColorRGBA type           
        if duration == None:
            self.mesh_marker.lifetime = rospy.Duration(0.0) # 0 = Marker never expires
        else:
            self.mesh_marker.lifetime = rospy.Duration(duration) # In seconds
        self.pub_rviz_marker.publish(self.mesh_marker)
