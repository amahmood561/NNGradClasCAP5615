import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class perceptron_learning_rule_question_8:
    def __init__(self):
        req1 =  [-1, 1, 0, 0, -1, 1]
        req2 = [-1, 1, -1, 1, 0, 0]
        req3 = [0, 1, 1, 0, 0, 1]
        req_weights = [1, 1, 1]
        # column names come from assignment requirement table
        column_labels = ["Round", "Input", 'Weight', 'V  calc', 'Desired', 'Actual', 'New Weight']
        self.final_output_table_dataframe = pd.DataFrame(columns=column_labels)
        # for iterator
        self.iteration_counter = 1
        # from requirement doc
        self.set_of_features = pd.DataFrame({"A": req1, "B": req2, "C":req3})
        # come frome assignment reqs
        self.initial_weights = np.array(req_weights)
        # learning rate given from assignment
        self.learning_rate = .5
        # rounds
        self.iteration_round = 2
        #get values
        self.label_value = self.set_of_features.C.values
        #get values get
        self.feat_value = self.set_of_features[["A", "B"]].values
        self.learning_rule()

    def calculate_v(self, transposed_value):
        # The np.dot () function accepts three arguments and returns the dot product of two given vectors
        return np.dot(self.initial_weights, transposed_value)

    def calculate_actual(self, value):
        # pass in v to calculate actual
        if (value >= 0):
            return 1
        else:
            return 0

    def calculate_d(self, i, current_actual):
        return (self.label_value[i] - current_actual)

    def calculate_feature_resolver(self, current_value, current_round, i):
        # transpose the data
        self.transposed_value = current_value.transpose()
        # calculate v column
        self.v_column_calculated = self.calculate_v(self.transposed_value)

        # calculate actual column
        self.calculatlated_actual_column = self.calculate_actual(self.v_column_calculated)
        # calc value
        self.current_d_calc = self.calculate_d(i, self.calculatlated_actual_column)
        # set weights to temp
        self.temp_weight = self.initial_weights

        if (self.current_d_calc != 0):
            calc_for_iw = (self.current_d_calc * current_value * self.learning_rate)
            self.initial_weights = calc_for_iw + self.initial_weights

        self.build_new_row(current_value, current_round, i)

    def build_new_row(self, current_value, current_round, i):
        # create the new row
        round = current_round + 1
        input = current_value
        column_w = self.temp_weight
        v_column = self.v_column_calculated
        desired = self.label_value[i]
        act = self.calculatlated_actual_column
        init_weights = self.initial_weights
        self.row = [round, input, column_w, v_column, desired, act, init_weights]
        self.add_to_grid()

    def add_to_grid(self):
        # add  row to data frame
        self.final_output_table_dataframe.loc[len(self.final_output_table_dataframe)] = self.row
        # add to iter counter
        self.iteration_counter += 1

    def learning_rule(self):
        for current_round in range(self.iteration_round):
            for index, current_value in enumerate(self.feat_value):
                current_value = np.insert(current_value, 0, 1)
                # handles each row on table
                self.calculate_feature_resolver(current_value, current_round, index)
        display(self.final_output_table_dataframe)


perceptron_learning_rule_question_8()