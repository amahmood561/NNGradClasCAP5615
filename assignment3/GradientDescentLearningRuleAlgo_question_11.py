import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class gradient_descent_learning_rule_question_11:
    def __init__(self, total_iterations, error_threshold_param):
        self.features = ''
        self.labels = ''
        self.Gd_calculated_learning_rate = ''
        self.test_features = ''
        self.test_labels = ''
        self.max_iterations = total_iterations
        self.err_threshold = error_threshold_param
        self.load_data_for_init()
        self.learning_rate = self.Gd_calculated_learning_rate
        self.Squared_Errors = []
        self.Squared_errors_validation = []
        self.current_ep = 0
        self.acc = []
        self.error_value = 9999.0
        self.current_error = 0
        self.randomWeight_calc = 0
        self.GradientDescentLearningHandler()
        self.color_gen = []
        self.colors = ["blue", "green", 'green']

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

    def load_data_for_init(self):
        # load data and train and set feats caluclate learning rate
        self.class1 = pd.read_csv("../Class1.txt")
        self.class2 = pd.read_csv("../Class2.txt")
        self.class_df_1 = self.class1.append(self.class2)
        self.class1.insert(self.class1.shape[1], 'label', 1)
        self.class_1_sample_df = self.class1.sample(frac=.8)
        self.class2.insert(self.class2.shape[1], 'label', -1)
        self.class_2_sample_df = self.class2.sample(frac=.8)
        self.class_df3 = self.class_1_sample_df.append(self.class_2_sample_df, sort=False)
        self.class_1_2_df_random = shuffle(self.class_df3)
        self.feats, labs = self.class_1_2_df_random.iloc[:, 0:-1], self.class_1_2_df_random.loc[:, ['label']]
        self.a_train, self.a_test, self.b_train, self.b_test = train_test_split(self.feats, labs, test_size=.2,
                                                                                random_state=45)
        self.a_train_m = np.asmatrix(self.a_train, dtype='float64')
        self.a_test_m = np.asmatrix(self.a_test, dtype='float64')
        self.b_train_m = np.asmatrix(self.b_train, dtype='float64')
        self.b_test_m = np.asmatrix(self.b_test, dtype='float64')
        self.features = self.a_train_m
        self.labels = self.b_train_m
        self.Gd_calculated_learning_rate = 1.0 / self.a_train.shape[0]
        self.test_features = self.a_test_m
        self.test_labels = self.b_test_m

    def gd_plotter_handler(self):
        rounds = np.arange(1, self.max_iterations + 1)
        plt.plot(rounds, self.Squared_Errors)
        plt.plot(rounds, self.Squared_errors_validation)
        plt.plot(rounds, self.acc)
        plt.xlabel('iters')
        plt.ylabel('mis_classified')
        plt.show()

    def handle_colors(self):
        self.color_gen = []
        for current_index in self.class_df3.iloc[:, 2]:
            self.color_gen.append(self.colors[current_index + 1])

    def scatter_plot_200_instances_requirement(self):
        self.handle_colors()
        print('Slope = ', (self.randomWeight_calc[0, 1] / self.randomWeight_calc[0, 2] * (-1)))
        print('Intercept = ', (self.randomWeight_calc[0, 0] / self.randomWeight_calc[0, 2] * (-1)))
        x_axis = self.class_df3.iloc[:, 0]
        y_axis = x_axis * (self.randomWeight_calc[0, 1] / self.randomWeight_calc[0, 2] * (-1)) + (
                    self.randomWeight_calc[0, 0] / self.randomWeight_calc[0, 2] * (-1))
        x = self.class_df3.iloc[:, 0]
        y = self.class_df3.iloc[:, 1]
        plt.scatter(x, y, color=self.color_gen)
        plt.plot(x_axis, y_axis, "g-")


gradient_object = gradient_descent_learning_rule_question_11(2000, 0.1)
print(gradient_object.Squared_Errors[0:15])
print(gradient_object.Squared_errors_validation[0:15])
print(gradient_object.acc[0:10])
gradient_object.gd_plotter_handler()


