#!/usr/bin/env python


import math
import rospy
import getpass
import numpy as np
from sklearn import svm as SVM
from sklearn.externals import joblib
from human_trajectory_classifier.svmc import SVMClassifier
from human_trajectory_classifier.trajectory_data import LabeledTrajectory


class TripleSVMClassifier(SVMClassifier):

    def __init__(self):
        self.stop_criteria = 0.01
        SVMClassifier.__init__(self)

    # loading the trained model if there is one
    def load_trained_model(self):
        try:
            self.svm = joblib.load(
                "/home/%s/STRANDS/semi_supervised_trajectory_classifier.pkl" % getpass.getuser()
            )
            rospy.loginfo("Using trained model semi_supervised_trajectory_classifier.pkl...")
            self.no_current_model = False
        except:
            rospy.loginfo("No trained model found...")
            self.no_current_model = True

    # updating model
    def update_model(self, training, label_train, unlabel):
        rospy.loginfo("Building a new trained model...")
        if not(0 in label_train and 1 in label_train):
            rospy.logwarn("Training data only contains one class.")
            self.no_current_model = True
        else:
            self._model_selection(training, label_train, unlabel)
            joblib.dump(
                self.svm,
                "/home/%s/STRANDS/semi_supervised_trajectory_classifier.pkl" % getpass.getuser()
            )
            self.no_current_model = False

    # choose the best C
    def _model_selection(self, training, label_train, unlabel):
        C = [0.01, 0.06, 0.21, 0.26, 0.51, 0.56, 0.71, 0.76, 1]
        fisher_ratio = list()
        for i in C:
            self.svm = SVM.LinearSVC(C=i)
            fisher_ratio.append(self._fit(training, label_train, unlabel))

        best_C = zip(C, fisher_ratio)
        best_C = sorted(zip(C, fisher_ratio), key=lambda i: i[1], reverse=True)
        rospy.loginfo("Best C is " + str(best_C[0]))
        self.svm = SVM.LinearSVC(C=best_C[0][0])
        self._fit(training, label_train, unlabel)

    # calculating fisher ratio
    def _calc_fisher_ratio(self, train_with_label):
        postive = list()
        negtive = list()
        for i in train_with_label:
            if i[1] == 1:
                postive.append(self.svm.decision_function(i[0]))
            else:
                negtive.append(self.svm.decision_function(i[0]))
        mean_pos = np.mean(postive)
        mean_neg = np.mean(negtive)
        var_pos = np.var(postive)
        var_neg = np.var(negtive)
        return (mean_pos - mean_neg)**2 / math.sqrt(var_pos * var_neg)

    # fitting the training data with label and print the result
    def _fit(self, training, label_train, unlabel, probability=False):
        rospy.loginfo("Fitting the training and unlabeled data...")
        fisher_ratios = list()
        self.svm.probability = probability
        print self.svm.fit(training, label_train)
        # classifying unlabel
        prediction = self.svm.predict(unlabel)
        # combine training with prediction of unlabeled, get the objective
        # function
        f_prev = self.calc_obj_func(
            zip(training, label_train) + zip(unlabel, prediction)
        )

        for i in range(15):
            # training
            print self.svm.fit(
                np.concatenate((training, unlabel), axis=0),
                np.concatenate((label_train, prediction), axis=0)
            )
            # calculate the objective function after training
            prediction = self.svm.predict(unlabel)
            f_post = self.calc_obj_func(
                zip(training, label_train) + zip(unlabel, prediction)
            )
            # stopping criteria
            print "Delta objective function is %f" % abs(f_post-f_prev)
            if abs(f_post - f_prev) < self.stop_criteria:
                break
            f_prev = f_post
            fisher_ratios.append(
                self._calc_fisher_ratio(
                    zip(training, label_train) + zip(unlabel, prediction)
                )
            )

        return max(fisher_ratios)

    # calculate the objective function of SVM
    # data is (train, label)
    def calc_obj_func(self, data):
        # calculating w in SVM equation
        # w = self.svm.dual_coef_[0].dot(self.svm.support_vectors_)
        w = math.sqrt(sum([i * i for i in self.svm.coef_[0]]))
        # calculate distance of the data from the margin
        sigma = 0
        for i in data:
            temp = 1
            if i[1] == 0:
                temp = -1
            hinge_loss = max(
                [0, 1 - (temp * self.svm.decision_function(i[0])[0])]
            )**2
            sigma += hinge_loss

        return ((w**2) / 2.0) + (self.svm.C * sigma)


if __name__ == '__main__':
    rospy.init_node("triplesvm_trajectory_classifier")

    trajs = LabeledTrajectory(update=False)
    trajs.get_data_from_file("/home/%s/test.data" % getpass.getuser())
    trajs.update_database()
    trajs.split_training_data()
    svmc = TripleSVMClassifier()
    if svmc.no_current_model:
        svmc.update_model(trajs.training, trajs.label_train, trajs.unlabel)
    svmc.calculate_accuracy(trajs.test, trajs.label_test)
