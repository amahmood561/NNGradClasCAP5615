import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
from IPython.display import display

class gradient_descent_learning_rule_question_9:
    def __init__(self):
        #given from requirements hw
        req1 =  [-1, 1, 0, 0, -1, 1]
        req2 =  [-1, 1, -1, 1, 0, 0]
        req3 =  [0, 1, 1, 0, 0, 1]
        req_weights = [1, 1, 1]
        # columns can be found from hw and video
        self.column_labels = ["round", "niput", 'Weight', 'vcalc', 'Desired', 'W', 'W_i', 'W__x', "Error", "W(k+1)"]
        self.final_output_table_dataframe = pd.DataFrame(columns=self.column_labels)
        self.features = pd.DataFrame({"a": req1, "b": req2, "c": req3})
        self.features["c"] = req3
        self.set_feats = self.features[["a", "b"]].values
        self.label_value = self.features.c.values
        self.list_of_errors = []
        self.init_weights = np.array(req_weights)
        self.learning_rate = .2
        self.iteration_round = 2
        self.iteration_counter = 1
        gd = self.learning_rule_gradient_descent()
        display(gd)

    def row_handler(self,round_,i,x,tempWeight,theta_w_summed):
        x = np.insert(x, 0, 1)
        v = np.dot(self.init_weights, x.transpose())
        calced_out = self.learning_rate * (self.label_value[i] - v)
        self.list_of_errors.append(self.label_value[i] - v)
        theta_w = calced_out * x
        theta_w_summed += theta_w
        self.build_row_columns(round_,tempWeight,theta_w,theta_w_summed,calced_out,i,x,v)
    def build_row_columns(self,round_, tempWeight,theta_w,theta_w_summed,calced_out,i,x,v):
        self.weight_list = []
        for _weight_ in tempWeight:
            self.weight_list.append("%2.1f" % _weight_)
        self.d_list = []
        for _del in theta_w:
            self.d_list.append("%2.1f" % _del)
        self.del_list = []
        for _delta_ in theta_w_summed:
            self.del_list.append("%2.1f" % _delta_)
        self.round_iter = round_ + 1
        self.input_ = x
        self.weight = self.weight_list
        self.v_cal = v
        self.desire_param = self.label_value[i]
        self.learning_d_n = calced_out
        self.d_weight = self.d_list
        self.d_w = np.array(self.del_list)
        self.row_to_add = [self.round_iter, self.input_,
                      self.weight,
                      self.v_cal,
                      self.desire_param,
                      self.learning_d_n,
                      self.d_weight,
                      self.d_w, '', '']
        self.add_to_grid()

    def add_to_grid(self):
        self.final_output_table_dataframe.loc[self.iteration_counter, self.column_labels] = self.row_to_add
        self.iteration_counter += 1

    def learning_rule_gradient_descent(self):
        for round_ in range(self.iteration_round):
            tempWeight = self.init_weights
            theta_w_summed = np.zeros(3)
            self.list_of_errors = []
            for count, value in enumerate( self.set_feats):
               self.row_handler(round_,count,value,tempWeight,theta_w_summed)
            self.init_weights = self.init_weights + theta_w_summed
            calc1 = (np.array(self.list_of_errors) ** 2.0).sum() / len( self.set_feats)
            list_init_we=[]
            for _w_ in self.init_weights:
                list_init_we.append("%2.3f" % _w_)
            final_col = [calc1,list_init_we]
            self.final_output_table_dataframe.loc[self.iteration_counter - 1, ["Error", "W(k+1)"]] = final_col
        display(self.final_output_table_dataframe)

gradient_descent_learning_rule_question_9()
