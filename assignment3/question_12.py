import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
import re
from sklearn.feature_extraction.text import CountVectorizer
from math import log2

class question_12_review100:
    def __init__(self):
        self.count_vectorizer = CountVectorizer()
        self.imported_review = pd.read_csv("../review100.csv")
        self.print_sentiment_review_movie_remove_dead()
        self.vectorize_data()
        self.parsee_feat_names_from_vectorizedd()
        self.create_df_for_most_common_Req()
        self.create_1000_freq_token()
        self.last_val = lambda x:  x[0] if len(x) != 0 else x
        self.information_gain_execution()

    def hand_column_row(self,current_r):
        #make regex faster throw it to lowercase and throw all patterns on list and add as you find them
        current_r = current_r.lower()
        parse_list = [r"<br />", r'(\n+)|(\s+)',r"[.,(){}]", r"[<>\\/]",r"\"",r" +"]
        for i in parse_list:
            current_r = re.sub(i, ' ', current_r)
        current_r = current_r.strip()
        return current_r

    def print_sentiment_review_movie_remove_dead(self):
        self.imported_review['movie review'] = self.imported_review['movie review'].apply(self.hand_column_row)

    def vectorize_data(self):
        values_to_transform = self.imported_review["movie review"].values
        self.vectorized = self.count_vectorizer.fit_transform(values_to_transform)

    def parsee_feat_names_from_vectorizedd(self):
        self.feature_names = self.count_vectorizer.get_feature_names()

    def create_df_for_most_common_Req(self):
        send_vectorized_to_list = self.vectorized.toarray()
        self.df_word_breakdown = pd.DataFrame(send_vectorized_to_list, columns=self.feature_names)
        self.most_frequent_work_req = self.df_word_breakdown.sum().sort_values(ascending=False)[:1000]

    def create_1000_freq_token(self):
        get_freq_word_with_req_from_hw = np.where(self.df_word_breakdown[self.most_frequent_work_req.index] > 0, 1, 0)
        get_index_from_most_freq = self.most_frequent_work_req.index
        self.features_1000_d_f = pd.DataFrame(get_freq_word_with_req_from_hw, columns=get_index_from_most_freq)
        self.features_1000_d_f["sentiment"] = self.imported_review["sentiment"]

    def entropy_handler(self, current_data_frame, i_col):
        current_calc_e = 0
        for i in current_data_frame[i_col].unique():
            len_current_df = len(current_data_frame)
            current_index_from_df = current_data_frame[i_col].value_counts()[i]
            a_calc = -(current_index_from_df / len_current_df)
            b_calc = log2(current_index_from_df / len_current_df)
            current_calc_e += a_calc * b_calc
        return current_calc_e

    def entrop_logic(self, i_col, sentiment_column):
        current = 0
        unique_from_df = self.features_1000_d_f[i_col].unique()
        total_calc_e = {}
        for i in unique_from_df:
            conditional_calculated = (self.features_1000_d_f[i_col].value_counts()[i] / len(self.features_1000_d_f))
            current_entropy_handler_calculated = self.entropy_handler(self.features_1000_d_f[self.features_1000_d_f[i_col] == i], sentiment_column)
            current += conditional_calculated * current_entropy_handler_calculated
            total_calc_e[i] = [current_entropy_handler_calculated]
        current_return = [current, total_calc_e]
        return current_return

    def ig_handler(self, i_column, sentiment_column):
        e_calced = self.entropy_handler(self.features_1000_d_f, sentiment_column)
        get_calc = self.entrop_logic(i_column, sentiment_column)
        first_ = self.last_val(get_calc)
        codition_e_calc = first_
        info_gain = e_calced - codition_e_calc
        row = [info_gain, codition_e_calc]
        return row

    def information_gain_execution(self):
        list_of_values = []
        curent_df = self.features_1000_d_f.columns[:-1]
        for i_column in curent_df:
            #runn it through information gain handler
            list_of_values.append(self.ig_handler(i_column, 'sentiment'))
        get_index_from_1000_df = self.features_1000_d_f.columns[:-1]
        self.information_gain = pd.DataFrame(list_of_values, columns=['infor_gain', "conditional_entropy"], index=get_index_from_1000_df)
        self.information_gain_top_50 = self.information_gain.sort_values(by="infor_gain", ascending=False)[:50]
        self.information_gain.sort_values(by="infor_gain", ascending=False)[:50]

    def plot_information_gaiin(self):
        plt.figure(figsize=(15, 5))
        plt.bar(self.information_gain_top_50.index, self.information_gain_top_50["infor_gain"], color='red', label="Words")
        plt.xticks(rotation=90)
        plt.xlabel("Words")
        plt.ylabel("Information Gain")
        plt.title("Top 50 Information Gain")
        plt.show()


q12_object = question_12_review100()