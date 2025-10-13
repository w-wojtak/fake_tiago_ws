#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 node that simulates a QR code detector (like ar_track_alvar).
- Publishes fake QR code detections as ar_track_alvar_msgs/AlvarMarkers.
- Publishes to /ar_pose_marker.
- Simulates object pickups by changing the pose of the markers at scheduled times.
- This node is designed to be a realistic replacement for the simple 
  fake_vision_publisher_node, allowing for better testing of bridge nodes.
"""

import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers, AlvarMarker
from geometry_msgs.msg import Pose, Point, Quaternion

class FakeARPublisher:
    def __init__(self):
        rospy.init_node('fake_ar_publisher_node', anonymous=True)

        # --- CONFIGURATION ---
        self.publish_topic = '/ar_pose_marker'
        self.publish_rate_hz = 5.0  # Publish at 5 Hz
        self.camera_frame_id = 'camera_rgb_optical_frame' # The frame poses are relative to

        # Mapping of object names to their QR code IDs
        self.object_to_qr_id = {
            'base': 10,
            'load': 25,
            'bearing': 42,
            'motor': 55
        }

        # Initial poses of objects "on the table" relative to the camera
        self.initial_poses = {
            'base':    Pose(position=Point(x=-0.3, y=0.1, z=0.8), orientation=Quaternion(w=1.0)),
            'load':    Pose(position=Point(x=-0.1, y=0.1, z=0.8), orientation=Quaternion(w=1.0)),
            'bearing': Pose(position=Point(x=0.1, y=0.1, z=0.8), orientation=Quaternion(w=1.0)),
            'motor':   Pose(position=Point(x=0.3, y=0.1, z=0.8), orientation=Quaternion(w=1.0))
        }
        
        # Pose of an object after it has been "picked up" (far away)
        self.picked_up_pose = Pose(position=Point(x=0.0, y=1.5, z=1.5), orientation=Quaternion(w=1.0))

        # Schedule for when each object is "picked up" (in wall-clock seconds)
        self.pickup_schedule = [
            ('base', 5.0),
            ('load', 10.0),
            ('bearing', 15.0),
            ('motor', 20.0),
        ]

        # --- STATE ---
        # This dictionary holds the current pose for each object. It will be modified over time.
        self.current_object_poses = self.initial_poses.copy()

        # --- ROS SETUP ---
        self.pub = rospy.Publisher(self.publish_topic, AlvarMarkers, queue_size=10)
        
        # Schedule the pickup events
        self.schedule_pickups()

        # Start the main publishing loop
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate_hz), self.publish_markers)
        
        rospy.loginfo("Fake AR Marker Publisher started.")
        rospy.loginfo(f"Publishing to '{self.publish_topic}' at {self.publish_rate_hz} Hz.")
        rospy.loginfo(f"Simulating poses in frame: '{self.camera_frame_id}'")

    def schedule_pickups(self):
        """Creates one-shot timers to trigger the pickup events."""
        for obj_name, pickup_time in self.pickup_schedule:
            rospy.loginfo(f"Scheduling pickup of '{obj_name}' (ID {self.object_to_qr_id[obj_name]}) at t={pickup_time:.1f}s")
            rospy.Timer(
                rospy.Duration(pickup_time),
                lambda event, o=obj_name: self.pickup_object(o),
                oneshot=True
            )

    def pickup_object(self, obj_name):
        """Simulates an object pickup by changing its pose."""
        if obj_name in self.current_object_poses:
            self.current_object_poses[obj_name] = self.picked_up_pose
            rospy.loginfo(f"PICKED UP: '{obj_name}' (ID {self.object_to_qr_id[obj_name]}) has been moved.")

    def publish_markers(self, event):
        """Constructs and publishes the AlvarMarkers message."""
        # Create the top-level message
        markers_msg = AlvarMarkers()
        markers_msg.header.stamp = rospy.Time.now()
        markers_msg.header.frame_id = self.camera_frame_id

        # Populate the list of markers based on their current poses
        for obj_name, current_pose in self.current_object_poses.items():
            marker = AlvarMarker()
            
            # Set the header and ID
            marker.header.stamp = markers_msg.header.stamp
            marker.header.frame_id = markers_msg.header.frame_id
            marker.id = self.object_to_qr_id[obj_name]
            
            # Set the pose (AlvarMarker uses a PoseStamped)
            marker.pose.header = marker.header
            marker.pose.pose = current_pose
            
            # Add a dummy confidence value
            marker.confidence = 0
            
            markers_msg.markers.append(marker)
        
        # Publish the message
        self.pub.publish(markers_msg)
        rospy.logdebug(f"Published {len(markers_msg.markers)} fake AR markers.")

def main():
    try:
        FakeARPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()