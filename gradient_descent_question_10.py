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
from IPython.display import display

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# todo port to flask
class gradient_descent_learning_rule_question_10_service:
    def __init__(self):

        self.column_labels = ["round", "Input", 'Weight', 'd_esired', 'V', 'output', 'deltaW', 'New Weight', "Error"]
        self.final_output_table_dataframe = pd.DataFrame(columns=self.column_labels)
        self.req1 = [-1, 1, 0, 0, -1, 1]
        self.req2 = [-1, 1, -1, 1, 0, 0]
        self.req3 = [0, 1, 1, 0, 0, 1]
        self.features = pd.DataFrame({
            "a": self.req1,
            "b": self.req2,
            "c": self.req3})
        self.features["c"] = self.req3
        self.set_feats = self.features[["a", "b"]].values
        self.lab_values = self.features.c.values
        self.error_value = 0
        self.init_weights = np.array([1, 1, 1])
        self.learning_rate = .2
        self.iteration_round = 1
        self.iteration_counter = 0
        gd = self.learning_rule_delta_handler()
        display(gd)

    def calculate_v(self, transposed_value):
        return np.dot(self.init_weights, transposed_value)

    def calculate_d(self, i, current_value, x):
        return self.learning_rate * (self.lab_values[i] - current_value) * x

    def learning_rate_delata_column_handler(self, calculated_d):
        lr_list = []
        for x in calculated_d:
            lr_list.append("%2.1f" % x)
        return lr_list

    def new_w_column_handler(self):
        new_weight_list = []
        for x in self.init_weights:
            new_weight_list.append("%.2f" % x)
        return new_weight_list

    def weight_column_calc(self, weight):
        new_weight_list = []
        for x in weight:
            new_weight_list.append("%2.1f" % x)
        return new_weight_list

    def row_handler(self, round_, x, i):
        transposed_value = x.transpose()
        self.v_calc = self.calculate_v(transposed_value)
        calculated_d = self.calculate_d(i, self.v_calc, x)
        self.cal_for_form = (self.lab_values[i] - self.v_calc) ** 2
        self.error_value = self.error_value + self.cal_for_form
        weight = self.init_weights
        self.init_weights = self.init_weights + calculated_d
        self.roundcolumn_val = round_ + 1
        weight_column_calc = np.array(self.weight_column_calc(weight))
        self.lr_calc = self.learning_rate_delata_column_handler(calculated_d)
        self.new_weight_column = self.new_w_column_handler()
        self.iteration_counter += 1
        self.build_row(x, i, weight_column_calc)

    def build_row(self, x, i, weight_column_calc):
        row = [self.roundcolumn_val,
               x,
               weight_column_calc,
               self.lab_values[i],
               self.v_calc,
               self.v_calc,
               self.lr_calc,
               self.new_weight_column,
               '']
        self.final_output_table_dataframe.loc[self.iteration_counter, self.column_labels] = row
        length_check = len(self.set_feats)
        current_validation = self.iteration_counter % length_check
        if current_validation == 0:
            calculate_E = self.error_value / len(self.set_feats)
            self.final_output_table_dataframe.loc[len(self.final_output_table_dataframe), "Error"] = calculate_E

    def learning_rule_delta_handler(self):
        for round in range(self.iteration_round):
            for count, value in enumerate(self.set_feats):
                value = np.insert(value, 0, 1)
                self.row_handler(round, value, count)
        return self.final_output_table_dataframe


gradient_descent_learning_rule_question_10_service()