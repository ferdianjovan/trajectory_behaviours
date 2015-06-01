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

    def __init__(self):
        self.accuracy = -1.0
        self.C = 0.1
        self.svm = SVM.SVC(cache_size=1000, C=0.1, gamma=10)
        self.no_current_model = False
        self.load_trained_model()

    # loading the trained model if there is one
    def load_trained_model(self):
        try:
            self.svm = joblib.load(
                "/home/%s/STRANDS/trajectory_classifier.pkl" % getpass.getuser()
            )
            rospy.loginfo("Using trained model trajectory_classifier.pkl...")
            self.no_current_model = False
        except:
            rospy.loginfo("No trained model found...")
            self.no_current_model = True

    # updating model
    def update_model(self, training, label_train):
        rospy.loginfo("Building a new trained model...")
        if not(0 in label_train and 1 in label_train):
            rospy.logwarn("Training data only contains one class. Skipping learning...")
            self.no_current_model = True
        else:
            self._fit(training, label_train)
            joblib.dump(
                self.svm, "/home/%s/STRANDS/trajectory_classifier.pkl" % getpass.getuser()
            )
            self.no_current_model = False

    # fitting the training data with label and print the result
    def _fit(self, training, label_train, probability=False):
        rospy.loginfo("Fitting the training data...")
        self.svm.probability = probability
        print self.svm.fit(training, label_train)

    # predict the class of the test data
    def predict_class_data(self, test):
        if self.no_current_model:
            rospy.logwarn("No model for this classifier, please provide training data")
            return None
        else:
            return self.svm.predict(test)

    # get the mean accuracy of the classifier
    def calculate_accuracy(self, test, label_test):
        rospy.loginfo("Getting the accuracy...")
        self.accuracy = self.svm.score(test, label_test) * 100
        rospy.loginfo("Accuracy is " + str(self.accuracy))

    # get tpr and tnr of several models (different gamma, and C parameters)
    def get_tpr_tnr(self, trajs):
        rospy.loginfo("Constructing tpr, tnr...")
        all_tpr = dict()
        all_tnr = dict()
        if len(trajs.test) == 0:
            trajs.split_training_data()
        # calculate accuracy of all combination of C and gamma
        for i in [0.1, 1, 10, 100, 1000]:
            temp = dict()
            temp2 = dict()
            for j in [0.01, 0.1, 1, 10, 100]:
                self.svm = SVM.SVC(cache_size=2000, C=i, gamma=j)
                self._fit(trajs.training, trajs.label_train)
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
        if len(trajs.test) == 0:
            trajs.split_training_data()

        # calculate accuracy of all combination of C and gamma
        for i in [0.1, 1, 10, 100, 1000]:
            for j in [0.01, 0.1, 1, 10, 100]:
                self.svm = SVM.SVC(cache_size=2000, C=i, gamma=j)
                self._fit(trajs.training, trajs.label_train, True)
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

    trajs = LabeledTrajectory(update=False)
    trajs.get_data_from_file("/home/%s/test.data" % getpass.getuser())
    trajs.update_database()
    trajs.split_training_data()
    svmc = SVMClassifier()
    if svmc.no_current_model:
        svmc.update_model(trajs.training, trajs.label_train)
    svmc.calculate_accuracy(trajs.test, trajs.label_test)
