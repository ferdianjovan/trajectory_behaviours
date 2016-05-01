#!/usr/bin/env python


import rospy
from strands_navigation_msgs.msg import TopologicalMap
from region_observation.util import robot_view_cone, get_soma_info
from periodic_poisson_processes.people_poisson import PoissonProcessesPeople
from region_observation.util import is_intersected, get_largest_intersected_regions
from strands_exploration_msgs.srv import GetExplorationTasks, GetExplorationTasksResponse


class PeopleCountingManager(object):

    def __init__(self, name):
        soma_config = rospy.get_param("~soma_config", "activity_exploration")
        time_window = rospy.get_param("~time_window", 10)
        time_increment = rospy.get_param("~time_increment", 1)
        periodic_cycle = rospy.get_param("~periodic_cycle", 10080)
        self.poisson_proc = PoissonProcessesPeople(
            soma_config, time_window, time_increment, periodic_cycle
        )
        self.poisson_proc.load_from_db()
        rospy.loginfo("Create a service %s/get_waypoints..." % name)
        self.service = rospy.Service(name+'/get_waypoints', GetExplorationTasks, self._srv_cb)
        self.topo_map = None
        self.region_wps = self._get_waypoints(soma_config)
        rospy.loginfo("Region ids and their nearest waypoints: %s" % str(self.region_wps))
        rospy.sleep(0.1)

    def spin(self):
        self.poisson_proc.continuous_update()
        rospy.spin()

    def _srv_cb(self, msg):
        rospy.loginfo(
            "Got a request to find waypoints to visit between %d and %d"
            % (msg.start_time.secs, msg.end_time.secs)
        )
        rates = self.poisson_proc.retrieve_from_to(
            msg.start_time, msg.end_time
        )
        result = list()
        for roi, poisson in rates:
            total_rate = sum(poisson.values())
            result.append((total_rate, roi))
        result = sorted(result, key=lambda i: i[0])
        total = float(sum([i[0] for i in result]))
        result = GetExplorationTasksResponse(
            map(lambda i: self.region_wps[i], [i[1] for i in result])[:3],
            map(lambda i: i/total, [i[0] for i in result])[:3]
        )
        print result
        return result

    def _topo_map_cb(self, topo_map):
        self.topo_map = topo_map

    def _get_waypoints(self, soma_config):
        region_wps = dict()
        # get regions information
        regions, _ = get_soma_info(soma_config)
        # get waypoint information
        topo_sub = rospy.Subscriber(
            "/topological_map", TopologicalMap, self._topo_map_cb, None, 10
        )
        while self.topo_map is None:
            rospy.sleep(0.1)
            rospy.logwarn("Trying to get information from /topological_map...")
        topo_sub.unregister()

        for wp in self.topo_map.nodes:
            wp_sight = robot_view_cone(wp.pose)
            rois = list()
            regions = list()
            for roi, region in regions.iteritems():
                if is_intersected(wp_sight, region):
                    regions.append(region)
                    rois.append(roi)
            if len(regions) > 1:
                rospy.logwarn(
                    "There are two or more regions covered by the sight of the robot in %s" % wp.name
                )
                rospy.loginfo("Trying to get the largest intersected area between robot's sight and regions")
                region = get_largest_intersected_regions(wp_sight, regions)
                roi = rois[regions.index(region)]
            elif len(regions) == 1:
                roi = rois[0]
            else:
                rospy.logwarn("No region is covered in this waypoint!")
                continue
            region_wps.update({roi: wp.name})
        return region_wps


if __name__ == '__main__':
    rospy.init_node("people_count_manager")
    pcm = PeopleCountingManager(rospy.get_name())
    pcm.spin()
