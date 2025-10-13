#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 node that translates object detections into DNF-compatible input matrices.
Publishes combined matrices on /dnf_inputs as Float32MultiArray messages.
Subscribes to /object_detections (std_msgs/String, JSON format).
"""

import rospy
import numpy as np
import json
from std_msgs.msg import Float32MultiArray, String

class VisionToDNF:
    def __init__(self):
        rospy.init_node('vision_to_dnf_node', anonymous=True)

        # Publisher & subscriber
        self.pub_inputs = rospy.Publisher('/dnf_inputs', Float32MultiArray, queue_size=10)
        self.sub_detections = rospy.Subscriber('/object_detections', String, self.detection_callback, queue_size=10)

        # Spatial and temporal grids
        self.x_lim, self.t_lim = 80, 15
        self.dx, self.dt = 0.2, 0.1
        self.x = np.arange(-self.x_lim, self.x_lim + self.dx, self.dx)
        self.t = np.arange(0, self.t_lim + self.dt, self.dt)

        # Gaussian parameters
        self.amplitude, self.width = 5.0, 2.0
        self.duration_steps = 10  # Each Gaussian lasts 10 timesteps

        # Object → Gaussian center mapping
        self.object_positions = {'base': -60, 'load': -20, 'bearing': 20, 'motor': 40}
        self.last_positions = {obj: None for obj in self.object_positions}
        self.movement_detected = set()

        # Input matrices (matrix1 and matrix2 are identical)
        n_t, n_x = len(self.t), len(self.x)
        self.input_matrix1 = np.zeros((n_t, n_x))
        self.input_matrix2 = np.zeros((n_t, n_x))
        self.input_matrix3 = np.zeros((n_t, n_x))

        # Publishing control
        self.current_idx = 0
        self.timer = rospy.Timer(rospy.Duration(1.0), self.publish_step)

        rospy.loginfo("Vision→DNF node started, listening to /object_detections.")

    # ----------------- Utility Methods -----------------

    def gaussian(self, center):
        return self.amplitude * np.exp(-((self.x - center) ** 2) / (2 * self.width ** 2))

    def check_movement(self, obj, new_x, new_y, threshold=0.05):
        last = self.last_positions[obj]
        if last is None:
            self.last_positions[obj] = {'x': new_x, 'y': new_y}
            return False
        dx, dy = abs(new_x - last['x']), abs(new_y - last['y'])
        if dx > threshold or dy > threshold:
            self.last_positions[obj] = {'x': new_x, 'y': new_y}
            return True
        return False

    # ----------------- Callbacks -----------------

    def detection_callback(self, msg):
        """Handle incoming JSON detections and update matrices."""
        try:
            data = json.loads(msg.data)
            detections = data.get('detections', [])

            for d in detections:
                obj = d.get('object')
                pos = d.get('position', {})
                x, y = pos.get('x', 0.0), pos.get('y', 0.0)

                if obj in self.object_positions and obj not in self.movement_detected:
                    if self.check_movement(obj, x, y):
                        rospy.loginfo(f"Movement detected for {obj}")
                        self.add_gaussian_to_future_steps(obj)
                        self.movement_detected.add(obj)

        except Exception as e:
            rospy.logerr(f"Error in detection_callback: {e}")

    # ----------------- Core Method -----------------

    def add_gaussian_to_future_steps(self, obj):
        """Add Gaussian to matrix1 and matrix2 over 10 future timesteps."""
        gaussian_profile = self.gaussian(self.object_positions[obj])
        start_idx = self.current_idx + int(2.0 / self.dt) if len(self.movement_detected) == 0 else self.current_idx
        start_idx = min(start_idx, len(self.t) - 1)
        end_idx = min(start_idx + self.duration_steps, len(self.t))

        for i in range(start_idx, end_idx):
            self.input_matrix1[i] += gaussian_profile
            self.input_matrix2[i] += gaussian_profile

        rospy.loginfo(f"Scheduled Gaussian for {obj} from t={self.t[start_idx]:.2f}s to t={self.t[end_idx-1]:.2f}s")

    # ----------------- Publishing -----------------

    def publish_step(self, event):
        """Publish combined input matrices for the current timestep."""
        if self.current_idx >= len(self.t):
            rospy.loginfo("Finished publishing all time steps.")
            self.timer.shutdown()
            return

        combined = np.concatenate([
            self.input_matrix1[self.current_idx],
            self.input_matrix2[self.current_idx],
            self.input_matrix3[self.current_idx]
        ])

        msg = Float32MultiArray(data=combined.tolist())
        self.pub_inputs.publish(msg)

        rospy.loginfo(f"Published t={self.t[self.current_idx]:.2f}s | max1={self.input_matrix1[self.current_idx].max():.2f}")

        self.current_idx += 1


def main():
    try:
        VisionToDNF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
