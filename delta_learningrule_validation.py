import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#todo port to flask
class gradient_descent_learning_rule_question_10_service:
    def __init__(self):

        column_labels =columns=["round", "Input", 'Weight', 'd', 'O', 'lr(d-O)Xi', 'New Weight', "Error"]
        self.final_output_table_dataframe = pd.DataFrame(columns=column_labels)

        self.features =pd.DataFrame({"a":[-1, 1, 0, 0, -1, 1],
                    "b":[-1, 1, -1, 1, 0, 0],
                    "c":[0, 1, 1, 0, 0, 1]})
        self.features["c"] = [0, 1, 1,0 ,0,1]
        self.set_feats = self.features[["a", "b"]].values
        self.lab_values = self.features.c.values
        self.error_value = 0
        self.init_weights = np.array([1, 1, 1])
        self.learning_rate = .2
        self.iteration_round = 1
        self.iteration_counter = 0
        gd = self.learning_rule_delta_handler()
        print(gd)
    def calculate_v(self, transposed_value):
        return np.dot(self.init_weights, transposed_value)

    def calculate_d(self, i, current_value, x):
        return self.learning_rate * (self.lab_values[i] - current_value) * x

    def row_handler(self, round, x, i):
        transposed_value = x.transpose()
        v = self.calculate_v(transposed_value)
        calculated_d = self.calculate_d(i, v, x)
        self.error_value = self.error_value + (self.lab_values[i] - v) ** 2
        w_ = self.init_weights
        self.init_weights = self.init_weights + calculated_d
        self.iteration_counter += 1
        row = [round + 1, x, np.array(
                ["%2.1f" % k for k in w_]), self.lab_values[i],
               v,
               ["%2.1f" % k for k in calculated_d],
               ["%.2f" % k for k in self.init_weights]]

        self.final_output_table_dataframe.loc[
            self.iteration_counter, ["round", "Input", 'Weight', 'd', 'O', 'lr(d-O)Xi', 'New Weight']] = row
        if (self.iteration_counter % len(self.set_feats) == 0):
            self.final_output_table_dataframe.loc[self.iteration_counter, "Error"] = self.error_value / len(self.set_feats)

    def learning_rule_delta_handler(self):
        for epoch in range(self.iteration_round):

            for i, x in enumerate(self.set_feats):
                x = np.insert(x, 0, 1)
                self.row_handler(epoch,x,i)
        return self.final_output_table_dataframe


gradient_descent_learning_rule_question_10_service()