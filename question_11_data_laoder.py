import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class questions_11_data_analysis:
    def __init__(self):
        self.colors = ["blue", "green", 'green']
        self.load_data_for_init()

    def load_data_for_init(self):
        #load data analyze print out and train
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
        self.features = self.a_train_m
        self.labels = self.b_train_m
        self.Gd_calculated_learning_rate = 1.0/self.a_train.shape[0]
        self.test_features = self.a_test_m
        self.test_labels = self.b_test_m
        current_index = 0
        self.data_example_ = []
        for i, row in self.class_df3.iterrows():
            current_index = current_index + 1
            if (((row['weight'] - row['height']) * (-row['label'])) >= 0.1):
                self.data_example_.append(current_index)
        print(self.data_example_)
    def handle_colors(self):
        self.color_gen = []
        for current_index in  self.class_df3.iloc[:, 2]:
            self.color_gen.append(self.colors[current_index + 1])
    def plot_data_80_requirement(self):
            plt.figure()
            plt.figure(figsize=(8, 7))
            x = self.class_df3.iloc[:, 0]
            y = self.class_df3.iloc[:, 1]
            self.handle_colors()
            plt.scatter(x, y, color=self.color_gen)
            plt.xlabel('Weight')
            plt.ylabel('Height')
            plt.title('80% instances the plot')
            plt.show()

q11_req_1 = questions_11_data_analysis()

q11_req_1.plot_data_80_requirement()

