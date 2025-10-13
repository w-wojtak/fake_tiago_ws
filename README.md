# DNF Sequential Learning Demo

This ROS package demonstrates a system that learns a sequence of object pickups using Dynamic Neural Fields (DNFs).

## Running the Learning Demo

The main launch file, `dnf_learning_pipeline.launch`, uses the `vision_source` argument to switch between different simulated vision inputs.

### Mode 1: `ar` (Mock QR Code Detector)

This is the primary mode for testing the full data pipeline. It simulates the output of a QR code detector (`ar_track_alvar`), which is then translated by the `robot_vision_bridge_node`. This is ideal for ensuring the integration between vision and the DNF system is working correctly.

`roslaunch fake_tiago_pkg dnf_learning_pipeline.launch vision_source:=ar`


###  Mode 2: simple (Direct JSON Publisher)
This is a minimal simulator for quick demos or for debugging the DNF system in isolation. It bypasses the bridge node and publishes the final JSON format directly, allowing you to test the DNF logic without any vision pipeline components.


`roslaunch fake_tiago_pkg dnf_learning_pipeline.launch vision_source:=simple`

### System Architecture
The data pipeline changes based on the selected mode:

ar mode pipeline:
fake_ar_publisher → /ar_pose_marker → robot_vision_bridge → /object_detections → vision_to_dnf

simple mode pipeline:
fake_vision_publisher → /object_detections → vision_to_dnf

### Integrating a Custom Vision System
To interface a real or custom vision system with this pipeline, two topics are used to connect the components:

/ar_pose_marker (Input to the Bridge):
A vision node (e.g., ar_track_alvar) should publish ar_track_alvar_msgs/AlvarMarkers to this topic. It is crucial that marker poses are in a stable, robot-centric coordinate frame (e.g., base_link).

/object_detections (Input to the DNF System):
The provided robot_vision_bridge_node automatically translates the marker data into the simple JSON format below. This is the final input consumed by the DNF system.

```JSON
{
  "timestamp": 123456.78,
  "detections": [
    {
      "object": "base",
      "position": {"x": -0.60, "y": 0.25, "z": 0.02}
    }
  ]
}
```