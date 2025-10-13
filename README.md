# DNF Sequential Learning System - Demo Scenarios

## Demo 1: Basic 4-Object Sequence

This demo shows how to use the DNF learning system with a simple 4-object assembly sequence (base → load → bearing → motor).

### Part 1: Learning Phase

**Launch**: `roslaunch fake_tiago_pkg dnf_basic_learning.launch`

**What happens**: The system observes 4 object detections in sequence and learns the temporal pattern using Dynamic Neural Fields.

**Nodes**:

1. **`fake_vision_publisher_node`** (for testing only)
   - Simulates object detections at scheduled times (t=2s, 5s, 8s, 11s)
   - Publishes to `/object_detections`
   - *Replace with TIAGo's vision in real experiments*

2. **`vision_to_dnf_node`** (bridge)
   - Converts detections → Gaussian inputs at fixed positions (base=-60, load=-20, bearing=20, motor=40)
   - Publishes to `/dnf_inputs` at 1 Hz (0.1s simulation timesteps)

3. **`dnf_model_learning_simple_node`** (learning core)
   - Runs two DNF fields: Sequence Memory (u_sm) + Task Duration (u_d)
   - Shows real-time visualization
   - Saves learned fields to `data_basic/` on completion (150s wall-time = 15s simulation)

**Expected result**: The sequence memory field learns the spatial-temporal pattern of the 4 objects.

---

### Part 2: Recall Phase

*[To be added]*

---

## Timing Notes

- **Wall-clock**: 1s between updates
- **Simulation**: 0.1s timesteps (10x slower than real-time)
- First detection at t=2.0s (simulation) = 20s (wall-clock)

## Using Real Vision

Replace `fake_vision_publisher_node` with TIAGo's vision. Required message format:

```json
{
  "detections": [
    {"object": "base", "position": {"x": -60.0, "y": 0.0}}
  ]
}