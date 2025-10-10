#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Header
import random

rospy.init_node('fake_right_hand')

pose_pub = rospy.Publisher('/dxl_input/pos_right', PoseStamped, queue_size=10)
gripper_pub = rospy.Publisher('/dxl_input/gripper_right', PointStamped, queue_size=10)

rate = rospy.Rate(1)  # 1 Hz
while not rospy.is_shutdown():
    pose_msg = PoseStamped()
    pose_msg.header = Header()
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.header.frame_id = "base_link"
    pose_msg.pose.position.x = random.random()
    pose_msg.pose.position.y = random.random()
    pose_msg.pose.position.z = random.random()
    pose_msg.pose.orientation.w = 1.0

    gripper_msg = PointStamped()
    gripper_msg.header = pose_msg.header
    gripper_msg.point.x = random.random()
    gripper_msg.point.y = random.random()
    gripper_msg.point.z = random.random()

    pose_pub.publish(pose_msg)
    gripper_pub.publish(gripper_msg)

    rate.sleep()
