#!/usr/bin/env python

import rospy
import math
import rosparam
from sklearn import cross_validation
from human_trajectory.trajectories import OfflineTrajectories

from mongodb_store.message_store import MessageStoreProxy
from human_trajectory_classifier.msg import TrajectoryType


class LabeledTrajectory(object):

    def __init__(self, chunk=20, update=True):
        self.chunk = chunk
        self.training = list()
        self.test = list()
        self.label_train = list()
        self.label_test = list()
        self.data_from_file = None
        self.data_from_mongo = None
        self.unlabel = list()

        if update:
            self.update_database()

    # get uuid and label (1 for positive class, 0 for negative class)
    def get_data_from_file(self, path):
        rospy.loginfo("Obtaining some data from file...")
        if path == '':
            rospy.logfatal("File does not exist, please specified the path correctly.")
            raise rospy.ROSException("Path to a file is wrong.")
            return
        self.data_from_file = rosparam.load_file(path)[0][0]

    # add training data based on data from file
    def _label_data_from_file(self, trajs):
        rospy.loginfo("Splitting data into chunks...")
        for uuid, traj in trajs.iteritems():
            label = 0
            chunked_traj = self.create_chunk(list(zip(*traj.humrobpose)[0]))
            if uuid in self.data_from_file:
                label = self.data_from_file[uuid]
                for i in chunked_traj:
                    self.training.append(i)
                    self.label_train.append(label)
            else:
                for i in chunked_traj:
                    self.unlabel.append(i)

    # get uuid and label (1 for positive class, 0 for negative class)
    # from mongodb
    def get_data_from_mongo(self):
        rospy.loginfo("Obtaining data from database...")
        self.data_from_mongo = dict()
        traj_types = MessageStoreProxy(collection="trajectory_types").query(
            TrajectoryType._type
        )
        for i in traj_types:
            self.data_from_mongo[i[0].uuid] = i[0].trajectory_type

    # add training data based on data from file
    def _label_data_from_mongo(self, trajs):
        rospy.loginfo("Splitting data into chunks...")
        for uuid, traj in trajs.iteritems():
            label = 0
            chunked_traj = self.create_chunk(list(zip(*traj.humrobpose)[0]))
            if uuid in self.data_from_mongo:
                if self.data_from_mongo[uuid] == 'human':
                    label = 1
                for i in chunked_traj:
                    self.training.append(i)
                    self.label_train.append(label)
            else:
                for i in chunked_traj:
                    self.unlabel.append(i)

    # update training data from database
    def update_database(self):
        rospy.loginfo("Updating database...")
        self.training = list()
        self.test = list()
        self.label_train = list()
        self.label_test = list()
        self.unlabel = list()
        if self.data_from_mongo is not None:
            self._label_data_from_mongo(OfflineTrajectories().traj)
        else:
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
            # labeling, positive class is 1, negative class is 0
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


if __name__ == '__main__':
    rospy.init_node("labeled_trajectory")
    lt = LabeledTrajectory(update=False)
    # lt.get_data_from_file("/home/fxj345/test.data")
    lt.get_data_from_mongo()
    lt.update_database()
    lt.split_training_data()
    print len(lt.label_train), len(lt.label_test)
