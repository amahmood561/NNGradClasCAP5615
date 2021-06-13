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
class gradient_descent_learning_rule_question_9:
    def __init__(self):

        column_labels = ["round", "Input", 'Weight', 'v', 'Desired', 'lr(d(n)-o(n))', 'Delta Wi', 'Delta W', "E", "W(k+1)"]
        self.final_output_table_dataframe = pd.DataFrame(columns=column_labels)

        self.features = pd.DataFrame({"a": [-1, 1, 0, 0, -1, 1], "b": [-1, 1, -1, 1, 0, 0], "c": [0, 1, 1, 0, 0, 1]})
        self.features["c"] = [0, 1, 1, 0, 0, 1]
        self.set_feats = self.features[["a", "b"]].values
        self.label_value = self.features.c.values
        self.list_of_errors = []
        self.init_weights = np.array([1, 1, 1])
        self.learning_rate = .2
        self.iteration_round = 2
        self.iteration_counter = 1
        gd = self.learning_rule_gradient_descent()
        print(gd)

    def learning_rule_gradient_descent(self):
        for round in range(self.iteration_round):
            tempWeight = self.init_weights
            theta_w_summed = np.zeros(3)
            self.list_of_errors = []
            for i, x in enumerate( self.set_feats):
                x = np.insert(x, 0, 1)
                v = np.dot(self.init_weights, x.transpose())
                calced_out = self.learning_rate * (self.label_value[i] - v)
                self.list_of_errors.append(self.label_value[i] - v)
                theta_w = calced_out * x
                theta_w_summed += theta_w
                row_to_add = [round + 1, x,
                              ["%2.1f" % k for k in tempWeight],
                              v,
                              self.label_value[i],
                              calced_out,
                              ["%2.1f" % k for k in theta_w],
                              np.array(["%2.1f" % k for k in theta_w_summed])]

                self.final_output_table_dataframe.loc[self.iteration_counter, ["round", "Input", 'Weight', 'v', 'Desired', 'lr(d(n)-o(n))', 'Delta Wi', 'Delta W']] = row_to_add

                self.iteration_counter += 1

            self.init_weights = self.init_weights + theta_w_summed
            final_col = [(np.array(self.list_of_errors) ** 2.0).sum() / len( self.set_feats),
                        ["%2.3f" % k for k in self.init_weights]]
            self.final_output_table_dataframe.loc[self.iteration_counter - 1, ["E", "W(k+1)"]] = final_col
        print(self.final_output_table_dataframe)

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
gradient_descent_learning_rule_question_9()
