import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.model_selection import train_test_split

class perceptron_learning_rule_question_8:
    def __init__(self):
        # column names come from assignment requirement table
        column_labels = ["Round", "Input", 'Weight', 'V', 'Desired', 'Actual', 'New Weight']
        self.final_output_table_dataframe = pd.DataFrame(columns=column_labels)
        #for iterator
        self.iteration_counter = 1
        #from requirement doc
        self.set_of_features = pd.DataFrame({"A":[-1, 1, 0, 0, -1, 1], "B":[-1, 1, -1, 1, 0, 0], "C":[0, 1, 1, 0, 0, 1]})
        #come frome assignment reqs
        self.initial_weights = np.array([1,1,1])
        #learning rate given from assignment
        self.learning_rate = .5
        #rounds
        self.iteration_round = 2
        self.label_value = self.set_of_features.C.values
        self.feat_value = self.set_of_features[["A", "B"]].values
        self.learning_rule()

    def calculate_v(self, transposed_value):
        return np.dot(self.initial_weights, transposed_value)

    def calculate_actual(self, v):
        if(v >= 0 ):
            return 1
        else:
            return 0

    def calculate_d(self, i, current_actual):
        return (self.label_value[i] - current_actual)

    def calculate_feature_resolver(self, x, current_round, i):
        transposed_value = x.transpose()
        # calculate v column
        v_column_calculated = self.calculate_v(transposed_value)
        # calculate actual column
        calculatlated_actual_column = self.calculate_actual(v_column_calculated)
        current_d_calc = self.calculate_d(i, calculatlated_actual_column)
        temp_weight = self.initial_weights
        if (current_d_calc != 0):
            calc_for_iw = (current_d_calc * x * self.learning_rate)
            self.initial_weights = calc_for_iw + self.initial_weights
        #create the new row
        row = [current_round + 1,
               x,
               temp_weight,
               v_column_calculated,
               self.label_value[i],
               calculatlated_actual_column,
               self.initial_weights]
        #add  row to data frame
        self.final_output_table_dataframe.loc[self.iteration_counter, :] = row
        self.iteration_counter += 1

    def learning_rule(self):
        #iterate through features and rounds
        for current_round in range(self.iteration_round):
            for index, current_value in enumerate(self.feat_value):
                current_value = np.insert(current_value, 0, 1)
                # handles each row on table
                self.calculate_feature_resolver(current_value, current_round, index)
        print(self.final_output_table_dataframe)



def perceptron_lr(features, labels, num_iter, learning_rate, w):
    results = pd.DataFrame(columns=["Epoch", "Input", 'Weight', 'v', 'Desired', 'Actual', 'New Weight'])
    j = 1

    for epoch in range(num_iter):

        for i, x in enumerate(features):

            x = np.insert(x, 0, 1)

            v = np.dot(w, x.transpose())

            actual = 1 if (v >= 0) else 0

            delta = (labels[i] - actual)

            w_ = w

            if (delta != 0):
                w = w + (delta * x * learning_rate)

            results.loc[j, :] = [epoch + 1, x, w_, v, labels[i], actual, w]
            j += 1

    return results


df = pd.DataFrame({"A":[-1, 1, 0, 0, -1, 1],
                    "B":[-1, 1, -1, 1, 0, 0],
                    "C":[0, 1, 1, 0, 0, 1]})

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
w = np.array([1,1,1])
df1 = perceptron_lr(df[["A", "B"]].values, df.C.values, 2, 0.5, w)
print(df1.to_string())

perceptron_learning_rule_question_8()