#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 node that simulates a vision system by publishing fake object detections.
Publishes to /object_detections (std_msgs/String, JSON format).
"""

import rospy
import json
from std_msgs.msg import String

class FakeVisionPublisher:
    def __init__(self):
        rospy.init_node('fake_vision_publisher_node', anonymous=True)
        self.pub = rospy.Publisher('/object_detections', String, queue_size=10)

        # Object list with approximate x-positions
        self.object_positions = ['base', 'load', 'bearing', 'motor']
        self.object_coords = {'base': -60, 'load': -20, 'bearing': 20, 'motor': 40}

        # Delay parameters
        self.publish_delay = rospy.get_param('~publish_delay', 3.0)  # seconds
        self.start_delay = rospy.get_param('~start_delay', 2.0)      # wait before first detection

        rospy.loginfo("Fake Vision Publisher node started.")
        rospy.Timer(rospy.Duration(1.0), self.step, oneshot=True)  # start after small delay

        self.step_idx = 0
        self.timer = None

    def step(self, event):
        # wait before first detection
        rospy.sleep(self.start_delay)
        self.publish_next_detection()

    def publish_next_detection(self):
        if self.step_idx >= len(self.object_positions):
            rospy.loginfo("All fake detections published. Node idle.")
            return

        obj = self.object_positions[self.step_idx]
        x_pos = self.object_coords[obj]

        detection_msg = {
            "detections": [
                {"object": obj, "position": {"x": x_pos, "y": 0.0}}
            ]
        }

        msg = String()
        msg.data = json.dumps(detection_msg)
        self.pub.publish(msg)

        rospy.loginfo(f"Published fake detection for '{obj}' at x={x_pos}")

        self.step_idx += 1

        # Schedule next detection
        if self.step_idx < len(self.object_positions):
            self.timer = rospy.Timer(rospy.Duration(self.publish_delay),
                                     lambda event: self.publish_next_detection(),
                                     oneshot=True)


def main():
    try:
        FakeVisionPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
