#!/usr/bin/env python

import time
import rospy
import datetime
# import argparse
import actionlib
import scipy.spatial.distance as distance_calc

from soma_msgs.msg import SOMAROIObject
from occurrence_learning.msg import OccurrenceRate
from tf.transformations import euler_from_quaternion
from strands_navigation_msgs.msg import TopologicalMap
from mongodb_store.message_store import MessageStoreProxy
from soma_geospatial_store.geospatial_store import GeoSpatialStoreProxy
from occurrence_learning.occurrence_learning_util import robot_view_cone
from occurrence_learning.trajectory_region_est import TrajectoryRegionEstimate
from occurrence_learning.region_observation_time import RegionObservationTimeManager
from occurrence_learning.occurrence_learning_util import trajectory_estimate_for_date
from occurrence_learning.trajectory_occurrence_freq import TrajectoryOccurrenceFrequencies
from occurrence_learning.msg import OccurrenceRateLearningResult, OccurrenceRateLearningAction
from occurrence_learning.srv import TrajectoryOccurrenceRate, TrajectoryOccurrenceRateResponse


class TrajectoryArrivalRateManager(object):

    def __init__(
        self, name, soma_map, soma_config, minute_interval,
        window_interval, collection="occurrence_rates"
    ):
        self.topo_map = None
        self.soma_map = soma_map
        self.soma_config = soma_config
        self.minute_interval = minute_interval
        self.window_interval = window_interval
        self.rotm = RegionObservationTimeManager(soma_map, soma_config)
        self.tre = TrajectoryRegionEstimate(soma_map, soma_config, minute_interval)
        self.tof = TrajectoryOccurrenceFrequencies(
            soma_map, soma_config, minute_interval=minute_interval,
            window_interval=window_interval
        )
        self.gs = GeoSpatialStoreProxy('geospatial_store', 'soma')

        rospy.loginfo("Connect to database collection %s..." % collection)
        self._ms = MessageStoreProxy(collection=collection)
        self._soma_db = MessageStoreProxy(collection="soma_roi")
        rospy.loginfo("Create a service %s/service..." % name)
        self.service = rospy.Service(name+'/service', TrajectoryOccurrenceRate, self.srv_cb)
        rospy.loginfo("Create an action server %s..." % name)
        self._as = actionlib.SimpleActionServer(
            name, OccurrenceRateLearningAction, execute_cb=self.execute, auto_start=False
        )
        self._as.start()

    def execute(self, goal):
        temp_start_time = rospy.Time.now()
        curr_date = datetime.datetime.fromtimestamp(rospy.Time.now().secs)
        # curr_date = datetime.datetime(2016, 3, 10, 0, 50)
        if curr_date.hour >= 0 and curr_date.hour < 8:
            curr_date = curr_date - datetime.timedelta(hours=24)
        curr_date = datetime.datetime(curr_date.year, curr_date.month, curr_date.day, 0, 0)
        prev_date = curr_date - datetime.timedelta(hours=24)
        self.rotm.calculate_region_observation_duration([prev_date, curr_date], self.minute_interval)
        self.rotm.store_to_mongo()
        if self._as.is_preempt_requested():
            self._as.set_preempted()
            return

        curr_traj_est = self.tre.estimate_trajectories_daily(
            [curr_date.day], curr_date.month, curr_date.year
        )
        curr_traj_est = trajectory_estimate_for_date(curr_traj_est, curr_date)
        prev_traj_est = self.tre.estimate_trajectories_daily(
            [prev_date.day], prev_date.month, prev_date.year
        )
        prev_traj_est = trajectory_estimate_for_date(prev_traj_est, prev_date)
        if self._as.is_preempt_requested():
            self._as.set_preempted()
            return

        self.tof.update_tof_daily(curr_traj_est, prev_traj_est, curr_date)
        self.tof.store_tof()
        self.rotm.reinit()
        self.tof.reinit()
        temp_end_time = rospy.Time.now()
        rospy.loginfo("Time needed to complete this is %d" % (temp_end_time - temp_start_time).secs)
        self._as.set_succeeded(OccurrenceRateLearningResult())

    def _get_occurrence_rate_logs(self, start_time, end_time):
        filtered_logs = list()
        if (end_time - start_time).total_seconds() <= self.window_interval * 60:
            end_time = start_time + datetime.timedelta(seconds=self.minute_interval * 60)
        if start_time.hour == end_time.hour and start_time.day == end_time.day:
            query = {
                "soma": self.soma_map, "soma_config": self.soma_config,
                "duration.secs": self.window_interval*60, "day": start_time.weekday(),
                "hour": start_time.hour,
                "minute": {"$gte": start_time.minute, "$lt": end_time.minute}
            }
            logs = self._ms.query(OccurrenceRate._type, message_query=query)
            filtered_logs = logs
        elif start_time.hour != end_time.hour and start_time.day == end_time.day:
            query = {
                "soma": self.soma_map, "soma_config": self.soma_config,
                "duration.secs": self.window_interval*60, "day": start_time.weekday(),
                "hour": {"$gte": start_time.hour, "$lte": end_time.hour},
            }
            logs = self._ms.query(OccurrenceRate._type, message_query=query)
            for log in logs:
                if log[0].hour == start_time.hour and log[0].minute >= start_time.minute:
                    filtered_logs.append(log)
                elif log[0].hour == end_time.hour and log[0].minute < end_time.minute:
                    filtered_logs.append(log)
                elif log[0].hour > start_time.hour and log[0].hour < end_time.hour:
                    filtered_logs.append(log)
        else:
            day = start_time.weekday()
            end_day = end_time.weekday()
            query = {
                "soma": self.soma_map, "soma_config": self.soma_config,
                "duration.secs": self.window_interval*60,
                "day": {"$gte": day, "$lte": end_day}
            }
            logs = self._ms.query(OccurrenceRate._type, message_query=query)
            for log in logs:
                if log[0].day == day and log[0].hour >= start_time.hour and log[0].minute >= start_time.minute:
                    filtered_logs.append(log)
                if log[0].day == end_day and log[0].hour <= end_time.hour and log[0].minute < end_time.minute:
                    filtered_logs.append(log)
                elif end_day > day:
                    if log[0].day > day and log[0].day < end_day:
                        filtered_logs.append(log)
                elif end_day < day:
                    if log[0].day > day:
                        filtered_logs.append(log)
                    elif log[0].day < end_day:
                        filtered_logs.append(log)

        return filtered_logs

    def _choose_proper_region_to_observe(self, rois, wp_point):
        accumulate_dist = dict()
        for roi in rois:
            query = {
                "map": self.soma_map, "config": self.soma_config,
                "roi_id": roi
            }
            logs = self._soma_db.query(SOMAROIObject._type, message_query=query)
            for log in logs:
                if roi not in accumulate_dist:
                    accumulate_dist[roi] = 0.0
                accumulate_dist[roi] = distance_calc.euclidean(
                    [wp_point.pose.position.x, wp_point.pose.position.y],
                    [log[0].pose.position.x, log[0].pose.position.y]
                )
        accumulate_dist = sorted(accumulate_dist, key=lambda i: accumulate_dist[i], reverse=True)
        return accumulate_dist[-1]

    def _get_waypoints(self, region_ids):
        waypoints = dict()
        topo_sub = rospy.Subscriber(
            "/topological_map", TopologicalMap, self._topo_map_cb, None, 10
        )
        while self.topo_map is None:
            rospy.sleep(0.1)
            rospy.logwarn("Trying to get information from /topological_map...")
        topo_sub.unregister()

        for wp in self.topo_map.nodes:
            if wp.name == "ChargingPoint":
                continue
            _, _, yaw = euler_from_quaternion(
                [0, 0, wp.pose.orientation.z, wp.pose.orientation.w]
            )
            coords = robot_view_cone(wp.pose.position.x, wp.pose.position.y, yaw)
            langitude_latitude = list()
            for pt in coords:
                langitude_latitude.append(self.gs.coords_to_lnglat(pt[0], pt[1]))
            langitude_latitude.append(self.gs.coords_to_lnglat(coords[0][0], coords[0][1]))

            if self.gs.observed_roi(langitude_latitude, self.soma_map, self.soma_config).count() > 1:
                rospy.logwarn("There are two or more regions covered by the sight of the robot in %s" % wp.name)
            roi = list()
            for j in self.gs.observed_roi(langitude_latitude, self.soma_map, self.soma_config):
                roi.append(str(j['soma_roi_id']))
            roi = [i for i in roi if i in region_ids]
            if len(roi) < 1:
                continue
            elif len(roi) == 1:
                waypoints[roi[0]] = wp.name
            else:
                roi = self._choose_proper_region_to_observe(roi, wp)
                waypoints[roi] = wp.name
        return waypoints

    def _topo_map_cb(self, topo_map):
        self.topo_map = topo_map

    def _get_long_duration(self, four_tuple, start_time):
        datetimes = list()
        for i in four_tuple:
            delta_day = (i[0] - start_time.weekday()) % 7
            curr_date = start_time + datetime.timedelta(days=delta_day)
            curr_date = datetime.datetime(
                curr_date.year, curr_date.month, curr_date.day, i[1], i[2]
            )
            datetimes.append(curr_date)
        datetimes = sorted(datetimes)
        counter = 0
        for i in range(1, len(datetimes)):
            if (datetimes[i] - datetimes[i - 1]).seconds == self.minute_interval * 60:
                counter += 1
            elif (datetimes[i] - datetimes[i - 1]).seconds <= self.window_interval * 60:
                counter += (datetimes[i] - datetimes[i - 1]).seconds / (self.minute_interval * 60)
        duration = rospy.Duration(
            self.window_interval * 60 + (counter * self.minute_interval * 60)
        )
        best_time = rospy.Time(time.mktime(datetimes[0].timetuple()))
        return best_time, duration

    def _get_best_regions(self, logs, start_time):
        or_regions = dict()
        best_time = dict()
        for log in logs:
            if log[0].region_id not in or_regions:
                or_regions[log[0].region_id] = 0.0
            or_regions[log[0].region_id] += log[0].occurrence_rate
            if log[0].region_id not in best_time:
                best_time[log[0].region_id] = [[
                    log[0].day, log[0].hour, log[0].minute, log[0].occurrence_rate
                ]]
            elif best_time[log[0].region_id][-1][-1] < log[0].occurrence_rate:
                best_time[log[0].region_id][-1] = [
                    log[0].day, log[0].hour, log[0].minute, log[0].occurrence_rate
                ]
            elif best_time[log[0].region_id][-1][-1] == log[0].occurrence_rate:
                best_time[log[0].region_id].append([
                    log[0].day, log[0].hour, log[0].minute, log[0].occurrence_rate
                ])

        duration = dict()
        for roi in best_time.keys():
            temp = best_time[roi]
            if len(temp) > 0:
                best_time[roi], duration[roi] = self._get_long_duration(temp, start_time)
            else:
                rospy.logwarn("There is no recommended visiting time for region %s" % roi)
                duration[roi] = rospy.Duration(0)
                best_time[roi] = rospy.Time.now()
        return or_regions, best_time, duration

    def srv_cb(self, msg):
        region_ids = list()
        best_times_list = list()
        durations_list = list()
        waypoints_list = list()
        rospy.loginfo("Got a request...")
        print msg
        rospy.loginfo("Retrieving trajectory occurrence frequencies from database...")
        start_time = datetime.datetime.fromtimestamp(msg.start_time.secs)
        end_time = datetime.datetime.fromtimestamp(msg.end_time.secs)
        if end_time >= start_time:
            logs = self._get_occurrence_rate_logs(start_time, end_time)
            # calculate occurrence_rate per region
            or_regions, best_times, durations = self._get_best_regions(logs, start_time)
            region_ids = sorted(or_regions, key=lambda i: or_regions[i], reverse=True)
            # get the nearest waypoints somehow
            waypoints = self._get_waypoints(region_ids)
            # ordering
            for roi in region_ids:
                best_times_list.append(best_times[roi])
                durations_list.append(durations[roi])
                waypoints_list.append(waypoints[roi])
            rospy.loginfo("Composing answer...")
            print TrajectoryOccurrenceRateResponse(
                region_ids[:msg.n_best], best_times_list[:msg.n_best],
                durations_list[:msg.n_best], waypoints_list[:msg.n_best]
            )
        return TrajectoryOccurrenceRateResponse(
            region_ids[:msg.n_best], best_times_list[:msg.n_best],
            durations_list[:msg.n_best], waypoints_list[:msg.n_best]
        )


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(prog="TrajectoryArrivalRate")
    # parser.add_argument('soma_map', help="Soma Map")
    # parser.add_argument('soma_config', help="Soma Config")
    # parser.add_argument('minute_interval', help="The Increment Minute Interval")
    # parser.add_argument('window_interval', help="Poisson Time Interval")
    # args = parser.parse_args()

    rospy.init_node("TrajectoryArrivalRate")
    TrajectoryArrivalRateManager(
        rospy.get_name(),
        rospy.get_param("~soma_map"), rospy.get_param("~soma_config"),
        rospy.get_param("~minute_increment", 1), rospy.get_param("~time_window", 10),
    )
    rospy.spin()
