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
from time import time

class VisionToDNF:
    def __init__(self):
        rospy.init_node('vision_to_dnf_node', anonymous=True)

        # Publisher & subscriber
        self.pub_inputs = rospy.Publisher('/dnf_inputs', Float32MultiArray, queue_size=10)
        self.sub_detections = rospy.Subscriber('/object_detections', String, self.detection_callback, queue_size=10)

        # Simulation parameters (must match DNF node)
        self.x_lim, self.t_lim = 80, 15
        self.dx, self.dt = 0.2, 0.1

        # Define spatial and temporal grids
        self.x = np.arange(-self.x_lim, self.x_lim + self.dx, self.dx)
        self.t = np.arange(0, self.t_lim + self.dt, self.dt)

        # Gaussian parameters
        self.amplitude = 5.0
        self.width = 2.0
        self.duration = 1.0  # 1 second duration

        # Object → Gaussian center mapping
        self.object_positions = {'base': -60, 'load': -20, 'bearing': 20, 'motor': 40}
        
        # Store last known positions for movement detection
        self.last_positions = {obj: None for obj in self.object_positions}
        self.movement_threshold = 0.05
        self.movement_detected = set()

        # Lists to store active gaussians for both matrices
        self.active_gaussians_matrix1 = []
        self.active_gaussians_matrix2 = []

        # Initialize input matrices
        self.input_matrix1 = np.zeros((len(self.t), len(self.x)))
        self.input_matrix2 = np.zeros((len(self.t), len(self.x)))
        self.input_matrix_3 = np.zeros((len(self.t), len(self.x)))

        # Initialize the current time index for publishing
        self.current_time_index = 0
        
        # Track actual elapsed time
        self.start_time = None

        # Timer to publish every 1 second
        self.timer = rospy.Timer(rospy.Duration(1.0), self.publish_slices)

        rospy.loginfo("Vision→DNF node started, listening to /object_detections.")

    def gaussian(self, center=0, amplitude=1.0, width=1.0):
        """Generate Gaussian profile centered at 'center'"""
        return amplitude * np.exp(-((self.x - center) ** 2) / (2 * (width ** 2)))

    def check_movement(self, object_name, new_x, new_y):
        """Check if object has moved beyond threshold"""
        if self.last_positions[object_name] is None:
            # First detection - treat as movement
            self.last_positions[object_name] = {'x': new_x, 'y': new_y}
            return True
        
        last_x = self.last_positions[object_name]['x']
        last_y = self.last_positions[object_name]['y']
        
        dx = abs(new_x - last_x)
        dy = abs(new_y - last_y)
        
        if dx > self.movement_threshold or dy > self.movement_threshold:
            self.last_positions[object_name] = {'x': new_x, 'y': new_y}
            return True
        
        return False

    def add_gaussian_input(self, object_name):
        """Add a gaussian input for the specified object to both matrices"""
        current_time = self.t[self.current_time_index]
        t_start = current_time
        t_stop = t_start + self.duration
        
        center = self.object_positions[object_name]
        
        gaussian_params = {
            'center': center,
            'amplitude': self.amplitude,
            'width': self.width,
            't_start': t_start,
            't_stop': t_stop
        }
        
        self.active_gaussians_matrix1.append(gaussian_params.copy())
        self.active_gaussians_matrix2.append(gaussian_params.copy())
        
        elapsed = time() - self.start_time if self.start_time else 0
        rospy.loginfo(f"Added gaussian for '{object_name}' at x={center}, sim_t={t_start:.1f}s (elapsed={elapsed:.1f}s)")

    def update_input_matrices(self):
        """Update both input matrices based on their active gaussians"""
        current_time = self.t[self.current_time_index]
        
        # Update matrix 1
        self.input_matrix1[self.current_time_index] = np.zeros(len(self.x))
        active_gaussians_copy1 = self.active_gaussians_matrix1.copy()
        
        for gaussian in active_gaussians_copy1:
            if gaussian['t_start'] <= current_time <= gaussian['t_stop']:
                self.input_matrix1[self.current_time_index] += self.gaussian(
                    center=gaussian['center'],
                    amplitude=gaussian['amplitude'],
                    width=gaussian['width']
                )
            elif current_time > gaussian['t_stop']:
                self.active_gaussians_matrix1.remove(gaussian)

        # Update matrix 2
        self.input_matrix2[self.current_time_index] = np.zeros(len(self.x))
        active_gaussians_copy2 = self.active_gaussians_matrix2.copy()
        
        for gaussian in active_gaussians_copy2:
            if gaussian['t_start'] <= current_time <= gaussian['t_stop']:
                self.input_matrix2[self.current_time_index] += self.gaussian(
                    center=gaussian['center'],
                    amplitude=gaussian['amplitude'],
                    width=gaussian['width']
                )
            elif current_time > gaussian['t_stop']:
                self.active_gaussians_matrix2.remove(gaussian)

    def detection_callback(self, msg):
        """Callback for object detection messages"""
        try:
            detection_data = json.loads(msg.data)
            detections = detection_data.get('detections', [])
            
            for detection in detections:
                object_name = detection.get('object', 'Unknown')
                position = detection.get('position', {})
                
                x = position.get('x', 0.0)
                y = position.get('y', 0.0)
                
                if object_name in self.object_positions:
                    if object_name not in self.movement_detected:
                        if self.check_movement(object_name, x, y):
                            elapsed = time() - self.start_time if self.start_time else 0
                            rospy.loginfo(f"Movement detected for '{object_name}' (elapsed={elapsed:.1f}s)")
                            self.add_gaussian_input(object_name)
                            self.movement_detected.add(object_name)
            
            self.update_input_matrices()
            
        except Exception as e:
            rospy.logerr(f"Error processing detection message: {e}")

    def publish_slices(self, event):
        """Publish combined input matrices for the current timestep"""
        # Initialize start time on first publish
        if self.start_time is None:
            self.start_time = time()
        
        if self.current_time_index < len(self.t):
            self.update_input_matrices()
            
            combined_input = [
                self.input_matrix1[self.current_time_index].tolist(),
                self.input_matrix2[self.current_time_index].tolist(),
                self.input_matrix_3[self.current_time_index].tolist()
            ]

            msg = Float32MultiArray()
            msg.data = [item for sublist in combined_input for item in sublist]
            self.pub_inputs.publish(msg)

            # Calculate elapsed time
            elapsed = time() - self.start_time
            sim_time = self.t[self.current_time_index]
            
            rospy.loginfo(
                f"Published [elapsed={elapsed:.1f}s, sim_t={sim_time:.1f}s] | "
                f"Max: m1={self.input_matrix1[self.current_time_index].max():.2f}, "
                f"m2={self.input_matrix2[self.current_time_index].max():.2f}"
            )

            self.current_time_index += 1
        else:
            rospy.loginfo("Completed publishing all time slices.")
            self.timer.shutdown()


def main():
    try:
        node = VisionToDNF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()