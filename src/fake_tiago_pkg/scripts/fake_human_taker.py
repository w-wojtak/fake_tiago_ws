#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

class FakeHumanTaker:
    def __init__(self):
        rospy.init_node('fake_human_taker', anonymous=True)

        # This schedule simulates WHEN the human is ready to take each object (in seconds from start)
        self.take_schedule = [
            ('base', 7.0),      # Human is ready for 'base' at t=7s
            ('load', 17.0),     # Human is ready for 'load' at t=17s
            ('bearing', 27.0),  # ...and so on
            ('motor', 37.0)
        ]

        self.pub = rospy.Publisher('/simulation/human_take_object', String, queue_size=10)
        rospy.loginfo("Fake Human Taker started. Scheduling 'take' events.")
        self.schedule_take_events()

    def schedule_take_events(self):
        for object_name, take_time in self.take_schedule:
            rospy.Timer(
                rospy.Duration(take_time),
                lambda event, o=object_name: self.publish_take_event(o),
                oneshot=True
            )

    def publish_take_event(self, object_name):
        rospy.loginfo(f"HUMAN (SIMULATED): Taking '{object_name}' from robot's gripper.")
        self.pub.publish(String(data=object_name))

if __name__ == '__main__':
    FakeHumanTaker()
    rospy.spin()