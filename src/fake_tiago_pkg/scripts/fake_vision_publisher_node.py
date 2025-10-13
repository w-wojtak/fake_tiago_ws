#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 node that simulates a vision system by publishing fake object detections.
Publishes to /object_detections (std_msgs/String, JSON format).
Detections are timed to arrive at specific simulation times.
"""

import rospy
import json
from std_msgs.msg import String

class FakeVisionPublisher:
    def __init__(self):
        rospy.init_node('fake_vision_publisher_node', anonymous=True)
        self.pub = rospy.Publisher('/object_detections', String, queue_size=10)

        # Object detection schedule: (object_name, simulation_time)
        # Simulation time is the time in the DNF simulation (dt=0.1s steps)
        self.detection_schedule = [
            ('base', 2.0),      # Detect 'base' at t=2.0s simulation time
            ('load', 5.0),      # Detect 'load' at t=5.0s simulation time
            ('bearing', 8.0),   # Detect 'bearing' at t=8.0s simulation time
            ('motor', 11.0),    # Detect 'motor' at t=11.0s simulation time
        ]

        # Object x-coordinates
        self.object_coords = {'base': -60, 'load': -20, 'bearing': 20, 'motor': 40}

        # Timing conversion: 1 second wall time = 0.1s simulation time
        # Therefore: wall_time = simulation_time * 10
        self.sim_to_wall_factor = 10.0

        # Add a small startup delay to ensure other nodes are ready
        self.startup_delay = rospy.get_param('~startup_delay', 2.0)

        rospy.loginfo("Fake Vision Publisher node started.")
        rospy.loginfo(f"Detection schedule (simulation time): {[(obj, t) for obj, t in self.detection_schedule]}")
        
        # Schedule all detections
        self.schedule_detections()

    def schedule_detections(self):
        """Schedule all detections based on simulation time"""
        for obj, sim_time in self.detection_schedule:
            # Convert simulation time to wall time
            wall_time = sim_time * self.sim_to_wall_factor + self.startup_delay
            
            rospy.loginfo(f"Scheduling '{obj}' detection at t={sim_time:.1f}s (wall time: {wall_time:.1f}s)")
            
            # Create timer for this detection
            rospy.Timer(
                rospy.Duration(wall_time),
                lambda event, o=obj, s=sim_time: self.publish_detection(o, s),
                oneshot=True
            )

    def publish_detection(self, obj, sim_time):
        """Publish a single detection"""
        x_pos = self.object_coords.get(obj, 0)

        detection_msg = {
            "detections": [
                {"object": obj, "position": {"x": x_pos, "y": 0.0}}
            ]
        }

        msg = String()
        msg.data = json.dumps(detection_msg)
        self.pub.publish(msg)

        rospy.loginfo(f"Published detection: '{obj}' at x={x_pos} (target sim_time={sim_time:.1f}s)")


def main():
    try:
        FakeVisionPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()