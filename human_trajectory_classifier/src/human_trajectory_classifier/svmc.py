#!/usr/bin/env python

import sys
import rospy
import getpass
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import svm as SVM
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
from human_trajectory_classifier.trajectory_data import LabeledTrajectory


class SVMClassifier(object):

    def __init__(self, trajs):
        self.accuracy = -1.0
        self.svm = SVM.SVC(cache_size=1000, C=0.1, gamma=10)
        self.no_model = False
        self.load_trained_model(trajs)

    # loading the trained model if there is one
    def load_trained_model(self, trajs):
        self.no_model = False
        try:
            self.svm = joblib.load(
                "/home/%s/STRANDS/trajectory_classifier.pkl" % getpass.getuser()
            )
            rospy.loginfo("Using trained model...")
        except:
            if not(0 in trajs.label_train and 1 in trajs.label_train):
            # if len(trajs.training) == 0:
                rospy.logwarn("No model for this classifier, please provide training data")
                self.no_model = True
            else:
                self.update_model(trajs)

    # updating model
    def update_model(self, trajs):
        rospy.loginfo("Building a new trained model...")
        self._fit(trajs)
        joblib.dump(
            self.svm, "/home/%s/STRANDS/trajectory_classifier.pkl" % getpass.getuser()
        )

    # fitting the training data with label and print the result
    def _fit(self, trajs, probability=False):
        rospy.loginfo("Fitting the training data...")
        self.svm.probability = probability
        print self.svm.fit(trajs.training, trajs.label_train)

    # predict the class of the test data
    def predict_class_data(self, test):
        if self.no_model:
            rospy.logwarn("No model for this classifier, please provide training data")
            return None
        else:
            return self.svm.predict(test)

    # get the mean accuracy of the classifier
    def calculate_accuracy(self, trajs):
        rospy.loginfo("Getting the accuracy...")
        trajs.split_training_data()
        self._fit(trajs)
        self.accuracy = self.svm.score(trajs.test, trajs.label_test) * 100
        rospy.loginfo("Accuracy is " + str(self.accuracy))
        self.load_trained_model(trajs)

    # get tpr and tnr of several models (different gamma, and C parameters)
    def get_tpr_tnr(self, trajs):
        rospy.loginfo("Constructing tpr, tnr...")
        all_tpr = dict()
        all_tnr = dict()
        trajs.split_training_data()
        # calculate accuracy of all combination of C and gamma
        for i in [0.1, 1, 10, 100, 1000]:
            temp = dict()
            temp2 = dict()
            for j in [0.01, 0.1, 1, 10, 100]:
                self.svm = SVM.SVC(cache_size=2000, C=i, gamma=j)
                self._fit(trajs)
                tp = 0
                fp = 0
                tn = 0
                fn = 0
                for k, v in enumerate(trajs.test):
                    prediction = self.predict_class_data(v)
                    if prediction[-1] == 1 and trajs.label_test[k] == 1:
                        tp += 1
                    elif prediction[-1] == 1 and trajs.label_test[k] == 0:
                        fp += 1
                    elif prediction[-1] == 0 and trajs.label_test[k] == 0:
                        tn += 1
                    else:
                        fn += 1
                tpr = tp / float(tp + fn)
                tnr = tn / float(fp + tn)
                print "C: %0.1f, Gamma:%0.2f, TPR: %0.5f, TNR: %0.5f" % (i, j, tpr, tnr)
                temp[j] = tpr
                temp2[j] = tnr
            all_tpr[i] = temp
            all_tnr[i] = temp2

        return all_tpr, all_tnr

    # produce roc curve by varying C and gamma
    def get_roc_curve(self, trajs):
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        trajs.split_training_data()

        # calculate accuracy of all combination of C and gamma
        for i in [0.1, 1, 10, 100, 1000]:
            for j in [0.01, 0.1, 1, 10, 100]:
                self.svm = SVM.SVC(cache_size=2000, C=i, gamma=j)
                self._fit(trajs, True)
                # predict with probability, to be fed to roc_curve
                prediction = self.svm.predict_proba(trajs.test)
                fpr, tpr, threshold = roc_curve(
                    trajs.label_test, prediction[:, 1]
                )
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                # plot the result
                plt.plot(
                    fpr, tpr, lw=1,
                    label='C=%0.2f, Gamma=%0.3f (area = %0.2f)' % (
                        i, j, roc_auc
                    )
                )

        # calculate the average roc value
        mean_tpr /= 25
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(
            mean_fpr, mean_tpr, 'k--',
            label='Mean ROC (area = %0.2f)' % mean_auc, lw=2
        )

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Human Trajectory')
        plt.legend(loc='lower right')
        plt.show()


if __name__ == '__main__':
    rospy.init_node("svm_trajectory_classifier")

    if len(sys.argv) < 2:
        rospy.logerr(
            "usage: classifier train_ratio"
        )
        sys.exit(2)

    trajs = LabeledTrajectory()
    svmc = SVMClassifier(trajs)
    svmc.update_model(trajs)
    svmc.calculate_accuracy(trajs)
    rospy.loginfo("The overall accuracy is " + str(svmc.accuracy))
