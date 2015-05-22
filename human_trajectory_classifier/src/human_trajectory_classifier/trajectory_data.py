#!/usr/bin/env python

import rospy
import math
from sklearn import cross_validation
from human_trajectory.trajectories import OfflineTrajectories


class LabeledTrajectory(object):

    def __init__(self, chunk=20, update=True):
        self.chunk = chunk
        self.training = list()
        self.test = list()
        self.label_train = list()
        self.label_test = list()
        if update:
            self.update_database()

    # update training and test data from database
    def update_database(self):
        rospy.loginfo("Updating database...")
        self.training = list()
        self.test = list()
        self.label_train = list()
        self.label_test = list()
        self._label_data(OfflineTrajectories().traj)

    # labeling data
    def _label_data(self, trajs):
        rospy.loginfo("Splitting data into chunk...")
        for uuid, traj in trajs.iteritems():
            chunked_traj = self.create_chunk(list(zip(*traj.humrobpose)[0]))
            # velocity restriction
            start = traj.humrobpose[0][0].header.stamp
            end = traj.humrobpose[-1][0].header.stamp
            delta = float((end-start).secs + 0.000000001 * (end-start).nsecs)
            if delta != 0.0:
                avg_vel = traj.length[-1] / delta
            else:
                avg_vel = 0.0
            # distance restriction
            dist = self.maximum_distance(list(zip(*traj.humrobpose)[0]))
            # labeling
            label = not(
                traj.length[-1] < 0.1 or avg_vel < 0.5 or avg_vel > 1.5 and dist < 1.0
            )
            for i in chunked_traj:
                self.training.append(i)
                self.label_train.append(label)

    # get maximum distance from initial pose of a trajectory
    # to a particular pose
    def maximum_distance(self, traj_msg):
        init_pose = traj_msg[0].pose.position
        dist = 0
        for i in traj_msg:
            pose = i.pose.position
            length = math.hypot((init_pose.x - pose.x), (init_pose.y - pose.y))
            if dist < length:
                dist = length
        return dist

    # chunk data for each trajectory, it returns a 1-D list containing
    # normalised x, y poses [x_1, y_1, x_2, y_2, ..., x_n, y_n]
    def create_chunk(self, poses):
        i = 0
        chunk_trajectory = list()
        while i < len(poses) - (self.chunk - 1):
            temp = list()
            for j in range(self.chunk):
                temp.append([
                    poses[i + j].pose.position.x,
                    poses[i + j].pose.position.y
                ])
            temp = self.get_normalized_poses(temp)
            normalized = list()
            for k in temp:
                normalized.append(k[0])
                normalized.append(k[1])
            chunk_trajectory.append(normalized)
            i += self.chunk

        return chunk_trajectory

    # normalize poses so that the first pose becomes (0,0)
    # and the second pose becomes the base for the axis
    # with tangen, cos and sin
    def get_normalized_poses(self, poses):
        dx = poses[1][0] - poses[0][0]
        dy = poses[1][1] - poses[0][1]
        if dx < 0.00001:
            dx = 0.00000000000000000001
        rad = math.atan(dy / dx)
        for i, j in enumerate(poses):
            if i > 0:
                dx = j[0] - poses[0][0]
                dy = j[1] - poses[0][1]
                if dx < 0.00001:
                    dx = 0.00000000000000000001
                rad2 = math.atan(dy / dx)
                delta_rad = rad2 - rad
                if rad2 == 0:
                    r = dx / math.cos(rad2)
                else:
                    r = dy / math.sin(rad2)
                poses[i][0] = r * math.cos(delta_rad)
                poses[i][1] = r * math.sin(delta_rad)

        poses[0][0] = poses[0][1] = 0
        return poses

    # splitting training data into training and test data
    def split_training_data(self, training_ratio=0.8):
        rospy.loginfo("Splitting data into test and training...")
        (a, b, c, d) = cross_validation.train_test_split(
            self.training, self.label_train,
            train_size=training_ratio, random_state=0
        )
        self.training = a
        self.test = b
        self.label_train = c
        self.label_test = d
