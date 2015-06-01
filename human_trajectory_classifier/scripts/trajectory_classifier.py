#! /usr/bin/env python

import rospy
import actionlib
from std_msgs.msg import Header
from human_trajectory.trajectory import Trajectory
from human_trajectory.msg import Trajectories
from human_trajectory_classifier.svmc import SVMClassifier
from human_trajectory_classifier.sssvm import TripleSVMClassifier
from human_trajectory_classifier.msg import TrajectoryClassificationAction
from human_trajectory_classifier.msg import TrajectoryClassificationResult
from human_trajectory_classifier.msg import HumanTrajectoryClassification
from human_trajectory_classifier.trajectory_data import LabeledTrajectory


class TrajectoryClassifierServer(object):

    def __init__(self, name, path='', classifier='svm'):
        self.seq = 1
        self._action_name = name
        self.file_path = path
        self.classifier_type = classifier
        self.label_trajs = LabeledTrajectory(update=False)
        if classifier == 'svm' or path == '':
            self.classifier = SVMClassifier()
        else:
            self.classifier = TripleSVMClassifier()
        if self.classifier.no_current_model:
            self.update_classifier_model()
        self.trajs = dict()
        self.filtered_uuid = dict()

        # Start server
        rospy.loginfo("%s is starting an action server", name)
        self._as = actionlib.SimpleActionServer(
            self._action_name,
            TrajectoryClassificationAction,
            execute_cb=self.execute,
            auto_start=False
        )
        self._as.start()
        self._pub = rospy.Publisher(
            self._action_name+'/detections', HumanTrajectoryClassification, queue_size=10
        )
        rospy.loginfo("%s is ready", name)

    # get trajectory data
    def traj_callback(self, msg):
        for i in msg.trajectories:
            if i.uuid not in self.filtered_uuid:
                if i.uuid in self.trajs:
                    traj = self.trajs[i.uuid]
                else:
                    traj = Trajectory(i.uuid)

                traj.humrobpose.extend(zip(i.trajectory, i.robot))
                traj._calc_length()
                traj.sequence_id = i.sequence_id
                self.trajs.update({i.uuid: traj})

        temp = [i.uuid for i in msg.trajectories]
        for uuid in self.trajs.keys():
            if uuid not in temp:
                del self.trajs[uuid]

    # predicting human trajectory type
    def _prediction(self, trajs):
        for i in trajs.values():
            human_counter = 0
            if len(i.humrobpose) >= 5 * 20:
                chunked_traj = self.label_trajs.create_chunk(
                    list(zip(*i.humrobpose)[0])
                )
                for j in chunked_traj:
                    if self._as.is_preempt_requested():
                        break
                    result = self.classifier.predict_class_data(j)
                    if result[-1] == 1:
                        human_counter += 1
                if self._as.is_preempt_requested():
                    rospy.loginfo("The online prediction is preempted")
                    break
                if len(chunked_traj) > 0:
                    conf = human_counter/float(len(chunked_traj))
                    human = 'human'
                    if conf < 0.5:
                        conf = 1.0 - conf
                        human = 'non-human'
                    self.filtered_uuid.update(
                        {
                            i.uuid: {'type': human, 'confidence': conf}
                        }
                    )
        self._publish()

    # publishing human trajectory type using HumanTrajectoryClassification
    # message
    def _publish(self):
        if len(self.filtered_uuid) == 0:
            header = Header(self.seq, rospy.Time.now(), '')
            self._pub.publish(
                HumanTrajectoryClassification(header, '', '', 0, -1)
            )
            self.seq += 1
            rospy.sleep(0.1)

        for k, v in self.filtered_uuid.items():
            if k in self.trajs:
                header = Header(self.seq, rospy.Time.now(), '')
                self._pub.publish(
                    HumanTrajectoryClassification(
                        header, k, v['type'], v['confidence'],
                        self.classifier.accuracy
                    )
                )
                self.seq += 1
                rospy.sleep(0.1)
            else:
                del self.filtered_uuid[k]

    def get_online_prediction(self):
        if self.classifier.no_current_model:
            rospy.logwarn("No model for the classifier")
            self._as.set_aborted()
            return
        # Subscribe to trajectory publisher
        rospy.loginfo(
            "%s is subscribing to human_trajectories/trajectories/batch",
            self._action_name
        )
        s = rospy.Subscriber(
            "human_trajectories/trajectories/batch", Trajectories,
            self.traj_callback, None, 30
        )

        while not self._as.is_preempt_requested():
            trajs = self.trajs
            self._prediction(trajs)
        self._as.set_preempted()
        s.unregister()

    # update classifier database
    def update_classifier_model(self):
        rospy.loginfo("%s is updating database", self._action_name)
        if self.classifier_type != 'svm' and self.file_path != '':
            self.label_trajs.get_data_from_file(self.file_path)
        self.label_trajs.update_database()
        self.label_trajs.split_training_data(training_ratio=0.9)
        if self.classifier_type != 'svm' and self.file_path != '':
            self.classifier.update_model(
                self.label_trajs.training,
                self.label_trajs.label_train,
                self.label_trajs.unlabel
            )
        else:
            self.classifier.update_model(self.label_trajs.training, self.label_trajs.label_train)
        rospy.loginfo("%s is ready", self._action_name)

    def calculate_accuracy(self):
        if self._as.is_preempt_requested():
            rospy.loginfo("The overall accuracy request is preempted")
            self._as.set_preempted()
        else:
            if not(0 in self.label_trajs.label_train and 1 in self.label_trajs.label_train):
                rospy.loginfo("No reference data test and training, updating data...")
                self.update_classifier_model()
            if 0 in self.label_trajs.label_train and 1 in self.label_trajs.label_train:
                self.classifier.calculate_accuracy(
                    self.label_trajs.test, self.label_trajs.label_test
                )
                self._as.set_succeeded(TrajectoryClassificationResult(True))
            else:
                rospy.loginfo("Still no reference test and training. Aborting calculation...")
                self._as.set_aborted()

    # execute call back for action server
    def execute(self, goal):
        if goal.request == 'update':
            self.update_classifier_model()
            self._as.set_succeeded(TrajectoryClassificationResult(True))
        elif goal.request == 'accuracy':
            self.calculate_accuracy()
        else:
            self.get_online_prediction()


if __name__ == '__main__':
    rospy.init_node("human_trajectory_classifier_server")
    classifier = rospy.get_param("~classifier", "svm")
    path = rospy.get_param("~data_path", "")
    sv = TrajectoryClassifierServer(rospy.get_name(), path, classifier)
    rospy.spin()
