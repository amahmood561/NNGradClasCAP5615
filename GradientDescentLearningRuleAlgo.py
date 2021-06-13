import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.model_selection import train_test_split


class gradient_descent_learning_rule_question_11:
    def __init__(self, selected_featues, selected_label, max_iterations, current_learning_rate, error_threshold_param,
                 features_to_test_param, feature_label_param):
        self.features = selected_featues
        self.labels = selected_label
        self.max_iterations = max_iterations
        self.learning_rate = current_learning_rate
        self.err_threshold = error_threshold_param
        self.test_features = features_to_test_param
        self.test_labels = feature_label_param
        self.Squared_Errors = []
        self.Squared_errors_validation = []
        self.current_ep = 0
        self.acc = []
        self.error_value = 9999.0
        self.current_error = 0
        self.randomWeight_calc = 0
        self.GradientDescentLearningHandler()

    def init_weight(self):
        return np.random.rand(self.features.shape[1] + 1) - 0.5

    def enum_features(self, random_weight):
        for i, x in enumerate(self.features):
            x = np.insert(x, 0, 1)
            v = np.dot(random_weight, x.transpose())
            self.current_error = self.current_error + (self.labels[i] - v) * (self.labels[i] - v)
        self.current_error = np.ndarray.item(self.current_error)
        self.current_error = self.current_error / 2.0
        self.error_value = self.current_error / self.features.shape[0]
        self.Squared_Errors.append(self.current_error / self.test_features.shape[0])

    def enum_test(self, random_weight):
        for i, x in enumerate(self.test_features):
            x = np.insert(x, 0, 1)
            v = np.dot(random_weight, x.transpose())
            self.current_error = self.current_error + (self.test_labels[i] - v) * (self.test_labels[i] - v)
        self.current_error = np.ndarray.item(self.current_error)
        self.current_error = self.current_error / 2.0
        self.Squared_errors_validation.append(self.current_error / self.test_features.shape[0])

    def enum_test_with_label_validation(self, random_weight):
        for i, x in enumerate(self.test_features):
            x = np.insert(x, 0, 1)
            v = np.dot(random_weight, x.transpose())
            if ((v >= 0 and self.test_labels[i] < 0) or (v < 0 and self.test_labels[i] >= 0)):
                self.current_error = self.current_error + 1

    def GradientDescentLearningHandler(self):
        random_weight = self.init_weight()
        while (self.current_ep < self.max_iterations) and (self.error_value > self.err_threshold):
            misclassified = 0
            deltaw = [0] * (self.features.shape[1] + 1)
            for i, x in enumerate(self.features):
                x = np.insert(x, 0, 1)
                v = np.dot(random_weight, x.transpose())
                diff = self.learning_rate * (self.labels[i] - v)
                deltaw = deltaw + diff * x
            random_weight = random_weight + deltaw
            self.current_error = 0
            self.enum_features(random_weight)
            self.current_error = 0
            self.enum_test(random_weight)
            self.current_error = 0
            self.enum_test_with_label_validation(random_weight)
            self.current_error = float(self.current_error)
            self.current_error = self.current_error / self.test_features.shape[0]
            self.acc.append(1 - self.current_error)
            self.current_ep = self.current_ep + 1
            self.randomWeight_calc = random_weight;
