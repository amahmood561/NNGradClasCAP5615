import random
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#todo port to falsk on free time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
class delta_method_requirements:
    def __init__(self,  total_iterations, current_learning_rate, error_threshold_param):
        self.current_features = ''
        self.current_labels = ''
        self.load_data_for_init()
        self.max_iterations = total_iterations
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
            self.current_error_val = self.current_error_val + (self.current_labels[i] - v) * (self.current_labels[i] - v)

    def v_handler(self, x):
        return np.dot( self.random_weight_init, x.transpose())

    def differential_handler(self, index, v_column_calc):
        return self.current_learning_rate * (self.current_labels[index] - v_column_calc)

    def delta_handler(self):
        while (self.round < self.max_iterations) and (self.error_max > self.error_threshold_param):
            self.current_d = [0] * (self.current_features.shape[1] + 1)
            i = random.randrange(self.current_features.shape[0])
            feat = self.current_features[i,]
            feat = np.insert(feat, 0, 1)
            v = self.v_handler(feat)
            self.differeenttial = self.differential_handler(i, v)
            self.current_d = self.current_d + self.differeenttial * feat
            self.enum_currentFeats()
            self.errorHandling()

    def load_data_for_init(self):
        #load data and train and set feats
        self.class1 = pd.read_csv("class1.txt")
        self.class2 = pd.read_csv("class2.txt")
        self.class_df_1 = self.class1.append(self.class2)
        self.class1.insert(self.class1.shape[1], 'label', 1)
        self.class_1_sample_df = self.class1.sample(frac=.8)
        self.class2.insert(self.class2.shape[1], 'label', -1)
        self.class_2_sample_df =self. class2.sample(frac=.8)
        self.class_df3 = self.class_1_sample_df.append(self.class_2_sample_df, sort=False)
        self.class_1_2_df_random = shuffle(self.class_df3)
        self.feats, labs = self.class_1_2_df_random.iloc[:, 0:-1], self.class_1_2_df_random.loc[:, ['label']]
        self.a_train, self.a_test, self.b_train, self.b_test = train_test_split(self.feats, labs, test_size=.2, random_state=45)
        self.a_train_m = np.asmatrix(self.a_train, dtype='float64')
        self.a_test_m = np.asmatrix(self.a_test, dtype='float64')
        self.b_train_m = np.asmatrix(self.b_train, dtype='float64')
        self.b_test_m = np.asmatrix(self.b_test, dtype='float64')
        self.current_features = self.a_train_m
        self.current_labels = self.b_train_m

    def delta_method_plot_handler(self):
        rounds = np.arange(1, self.max_iterations+1)
        plt.plot(rounds, self.Squared_Errors)
        plt.xlabel('iterations')
        plt.ylabel('misclassified')
        plt.show()

delta = delta_method_requirements(2000, 0.01, 0.1)
print(delta.Squared_Errors[0:15])