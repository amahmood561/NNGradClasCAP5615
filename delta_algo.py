import random
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
class delta_method_requirements:
    def __init__(self, selected_featues, selected_label, max_iterations, current_learning_rate, error_threshold_param):
        self.current_features = selected_featues
        self.current_labels = selected_label
        self.max_iterations = max_iterations
        self.current_learning_rate = current_learning_rate
        self.error_threshold_param = error_threshold_param
        self.Squared_Errors = []
        self.Squared_errors_validation = []
        self.round = 0
        self.differeenttial = 0
        self.current_d = 0
        self.error_max = 9999.0
        self.random_weight_init = np.random.rand(self.current_features.shape[1] + 1) - 0.5
        self.delta_handler()
        self.current_error_val = 0
        self.delta_handler()
    def errorHandling(self):
        self.current_error_val = np.asscalar(self.current_error_val)
        self.current_error_val = self.current_error_val / 2.0
        self.Squared_Errors.append(self.current_error_val)
        self.error_max = self.current_error_val / self.current_features.shape[0]
        self.round = self.round + 1
    def enum_currentFeats(self):
        self.random_weight_init = self.random_weight_init + self.current_d
        self.current_error_val = 0
        for i, x in enumerate(self.current_features):
            x = np.insert(x, 0, 1)
            v = np.dot(self.random_weight_init, x.transpose())
            self.current_error_val = self.current_error_val + (self.current_labels[i] - v) * (
                        self.current_labels[i] - v)
    def delta_handler(self):
        while (self.round < self.max_iterations) and (self.error_max > self.error_threshold_param):
            misclassified = 0
            self.current_d = [0] * (self.current_features.shape[1] + 1)
            i = random.randrange(self.current_features.shape[0])
            x = self.current_features[i,]
            x = np.insert(x, 0, 1)
            v = np.dot( self.random_weight_init, x.transpose())
            self.differeenttial = self.current_learning_rate * (self.current_labels[i] - v)
            self.current_d = self.current_d + self.differeenttial * x
            self.enum_currentFeats()
            self.errorHandling()
