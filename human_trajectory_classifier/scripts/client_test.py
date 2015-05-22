#! /usr/bin/env python

import rospy
import actionlib
from human_trajectory_classifier.msg import TrajectoryClassificationAction
from human_trajectory_classifier.msg import TrajectoryClassificationGoal


def test():
    client = actionlib.SimpleActionClient(
        "human_trajectory_classification_server", TrajectoryClassificationAction
    )
    rospy.loginfo("waiting for server...")
    client.wait_for_server()

    while not rospy.is_shutdown():
        goal = TrajectoryClassificationGoal()
        request = raw_input("[update | accuracy | online | preempt]\n")
        if request != 'preempt':
            goal.request = request
            client.send_goal(goal)
            if request == 'update':
                client.wait_for_result()
                result = client.get_result()
                rospy.loginfo(str(result.updated))
        else:
            client.cancel_goal()


if __name__ == '__main__':
    rospy.init_node("human_movement_detection_tester")
    test()
    rospy.spin()
