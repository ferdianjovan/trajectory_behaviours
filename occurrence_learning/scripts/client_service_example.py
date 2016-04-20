#!/usr/bin/env python

import time
import rospy
import datetime
from occurrence_learning.srv import TrajectoryOccurrenceRate


def service_example():
    rospy.init_node("client_occurrence_learning_example")
    start_time = datetime.datetime(2016, 4, 20, 0, 0)
    end_time = datetime.datetime(2016, 4, 21, 0, 0)
    st_epoch = rospy.Time(time.mktime(start_time.timetuple()))
    et_epoch = rospy.Time(time.mktime(end_time.timetuple()))
    occurrence_rate = rospy.ServiceProxy(
        "/TrajectoryArrivalRate/service", TrajectoryOccurrenceRate
    )
    tor = occurrence_rate(st_epoch, et_epoch, 3)
    print tor
    for ind, roi in enumerate(tor.region_ids):
        visit_time = datetime.datetime.fromtimestamp(tor.visit_times[ind].secs)
        rospy.loginfo(
            "Region %s is better to be visited at %s" % (roi, str(visit_time))
        )


service_example()
