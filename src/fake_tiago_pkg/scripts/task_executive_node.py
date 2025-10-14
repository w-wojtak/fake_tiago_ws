#!/usr/bin/env python3
import rospy
import threading
from std_msgs.msg import Header, String, Float32MultiArray
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, PointStamped

class TaskExecutiveNode:
    def __init__(self):
        rospy.init_node('task_executive_node')

        # --- State Machine ---
        self.state = 'IDLE'  # IDLE, PICKING, WAITING_FOR_TAKE, RELEASING
        self.object_in_gripper = None
        self._lock = threading.Lock()

        # --- Mappings & Poses ---
        self.DNF_POS_TO_OBJECT = {-60.0: 'base', -20.0: 'load', 20.0: 'bearing', 40.0: 'motor'}
        self.OBJECT_TO_WORLD_POSE = {
            'base': Point(x=0.4, y=-0.2, z=0.1), 'load': Point(x=0.4, y=0.0, z=0.1),
            'bearing': Point(x=0.4, y=0.2, z=0.1), 'motor': Point(x=0.4, y=0.4, z=0.1)
        }
        self.handover_pose = PoseStamped(header=Header(frame_id="base_link"),
                                         pose=Pose(position=Point(x=0.3, y=0.1, z=0.3), orientation=Quaternion(w=1.0)))
        self.retract_pose = self.handover_pose # For simplicity, retract to the same pose

        # --- ROS Publishers & Subscribers ---
        self.arm_pub = rospy.Publisher('/dxl_input/pos_right', PoseStamped, queue_size=10)
        self.gripper_pub = rospy.Publisher('/dxl_input/gripper_right', PointStamped, queue_size=10)
        self.pickup_announcement_pub = rospy.Publisher('/simulation/robot_pickup', String, queue_size=10)
        
        rospy.Subscriber('threshold_crossings', Float32MultiArray, self.dnf_prediction_callback)
        rospy.Subscriber('/simulation/human_take_object', String, self.human_take_callback)
        
        rospy.loginfo("Stateful Task Executive started.")
        rospy.sleep(1.0)

    def set_state(self, new_state):
        with self._lock:
            if self.state != new_state:
                rospy.loginfo(f"STATE CHANGE: {self.state} -> {new_state}")
                self.state = new_state

    def dnf_prediction_callback(self, msg):
        with self._lock:
            if self.state != 'IDLE' or not msg.data: return
            
            dnf_pos = msg.data[0]
            object_name = self.DNF_POS_TO_OBJECT.get(dnf_pos)
            if object_name:
                self.object_in_gripper = object_name
                self.set_state('PICKING')
                threading.Thread(target=self.handle_picking_state).start()

    def human_take_callback(self, msg):
        with self._lock:
            if self.state != 'WAITING_FOR_TAKE': return

            human_take_time = rospy.Time.now()
            taken_object = msg.data
            
            if taken_object == self.object_in_gripper:
                print(f"METRIC: t_human_take = {human_take_time.to_sec()}") # <<< KEY METRIC
                self.set_state('RELEASING')
                threading.Thread(target=self.handle_releasing_state).start()
            else:
                rospy.logerr(f"SEQUENCE ERROR: Human took '{taken_object}' but robot was holding '{self.object_in_gripper}'!")

    def handle_picking_state(self):
        obj_name = self.object_in_gripper
        target_point = self.OBJECT_TO_WORLD_POSE.get(obj_name)
        rospy.loginfo(f"--- PICKING '{obj_name.upper()}' ---")
        
        pre_grasp_pose = PoseStamped(header=Header(frame_id="base_link"), pose=Pose(position=Point(target_point.x, target_point.y, target_point.z + 0.15), orientation=Quaternion(w=1.0)))
        self.arm_pub.publish(pre_grasp_pose); rospy.sleep(3.0)
        
        self.gripper_pub.publish(PointStamped(header=Header(frame_id="base_link"), point=Point(x=1.0))); rospy.sleep(1.0)
        
        grasp_pose = PoseStamped(header=Header(frame_id="base_link"), pose=Pose(position=target_point, orientation=Quaternion(w=1.0)))
        self.arm_pub.publish(grasp_pose); rospy.sleep(3.0)
        
        self.gripper_pub.publish(PointStamped(header=Header(frame_id="base_link"), point=Point(x=0.0))); rospy.sleep(1.0)

        # Announce to the vision simulator that the object is now gone from the table
        self.pickup_announcement_pub.publish(String(data=obj_name))

        self.arm_pub.publish(self.handover_pose)
        rospy.sleep(3.0)
        
        robot_ready_time = rospy.Time.now()
        print(f"METRIC: t_robot_ready = {robot_ready_time.to_sec()} with object '{obj_name}'") # <<< KEY METRIC
        
        self.set_state('WAITING_FOR_TAKE')

    def handle_releasing_state(self):
        rospy.loginfo(f"--- RELEASING '{self.object_in_gripper.upper()}' ---")
        
        # Open the gripper to release the object
        self.gripper_pub.publish(PointStamped(header=Header(frame_id="base_link"), point=Point(x=1.0)))
        handover_done_time = rospy.Time.now()
        rospy.sleep(1.0)
        
        print(f"METRIC: t_handover_done = {handover_done_time.to_sec()}") # <<< KEY METRIC
        
        # Retract arm to a safe position
        self.arm_pub.publish(self.retract_pose); rospy.sleep(3.0)
        
        self.object_in_gripper = None
        self.set_state('IDLE')
        rospy.loginfo("--- HANDOVER COMPLETE. ROBOT IS IDLE ---")

if __name__ == '__main__':
    TaskExecutiveNode()
    rospy.spin()