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
        self.imported_review = pd.read_csv("../final/review100.csv")
        self.print_sentiment_review_movie_remove_dead()
        self.vectorize_data()
        self.parsee_feat_names_from_vectorizedd()
        self.create_df_for_most_common_Req()
        self.create_1000_freq_token()

    def hand_column_row(row):
        row = row.lower()
        row = re.sub(r"<br />", ' ', row)
        row = re.sub(r'(\n+)|(\s+)', ' ', row)
        row = re.sub(r"[.,(){}]", ' ', row)
        row = re.sub(r"[.,(){}]", ' ', row)
        row = re.sub(r"[<>\\/]", ' ', row)
        row = re.sub(r"\"", ' ', row)
        row = re.sub(r" +", ' ', row)
        row = row.strip()
        return row

    def print_sentiment_review_movie_remove_dead(self):
        self.imported_review['movie review'] = self.imported_review['movie review'].apply(self.hand_column_row)

    def vectorize_data(self):
        self.vectorized = self.count_vectorizer.fit_transform(self.imported_review["movie review"].values)

    def parsee_feat_names_from_vectorizedd(self):
        self.feature_names = self.vectorized.get_feature_names()
        print(self.feature_names)

    def create_df_for_most_common_Req(self):
        send_vectorized_to_list = self.vectorized.toarray()
        self.df_bow = pd.DataFrame(send_vectorized_to_list, columns=self.feature_names)
        self.most_common = self.df_bow.sum().sort_values(ascending=False)[:1000]

    def create_1000_freq_token(self):
        self.df_1000 = pd.DataFrame(np.where(self.df_bow[self.most_common.index] > 0, 1, 0), columns=self.most_common.index)
        self.df_1000["sentiment"] = self.imported_review["sentiment"]

    def entropy_handler(self, df, column):
        aggr = 0
        for i in df[column].unique():
            aggr += -(df[column].value_counts()[i] / len(df)) * log2(df[column].value_counts()[i] / len(df))
        return aggr

    def ig_handler(self, df, column, syscolumn):
        def conditional_entropy():
            aggr = 0
            entropies = {}
            for i in df[column].unique():
                aggr += (df[column].value_counts()[i] / len(df)) * entropy(df[df[column] == i], syscolumn)
                entropies[i] = [entropy(df[df[column] == i], syscolumn)]
            return [aggr, entropies]

        return [self.entropy(df, syscolumn) - conditional_entropy()[0], conditional_entropy()[0]]

    def information_gain_execution(self):
        self.ig = pd.DataFrame([information_gain(self.df_1000, i, 'sentiment') for i in self.df_1000.columns[:-1]],
                          columns=['IG', "CE"], index=self.df_1000.columns[:-1])
        self.ig50 = self.ig.sort_values(by = "IG", ascending=False)[:50]

    def plot_information_gaiin(self):
        plt.figure(figsize=(15, 5))
        plt.bar(self.ig50.index, ig50["IG"], color='red', label="Words")
        plt.xticks(rotation=90)
        plt.xlabel("Words")
        plt.ylabel("Information Gain")
        plt.title("Top 50 Information Gain")
        plt.show()
    def sort_info_gain_handler(self):
        ig.sort_values(by="IG", ascending=False)[:50]


from math import log2


def entropy(df, column):
    aggr = 0
    for i in df[column].unique():
        aggr += -(df[column].value_counts()[i] / len(df)) * log2(df[column].value_counts()[i] / len(df))
    return aggr


def information_gain(df, column, syscolumn):
    def conditional_entropy():
        aggr = 0
        entropies = {}
        for i in df[column].unique():
            aggr += (df[column].value_counts()[i] / len(df)) * entropy(df[df[column] == i], syscolumn)
            entropies[i] = [entropy(df[df[column] == i], syscolumn)]
        return [aggr, entropies]

    return [entropy(df, syscolumn) - conditional_entropy()[0], conditional_entropy()[0]]



ig = pd.DataFrame([information_gain(df_1000, i, 'sentiment') for i in df_1000.columns[:-1]], columns = ['IG', "CE"], index =df_1000.columns[:-1])


ig50 = ig.sort_values(by = "IG", ascending=False)[:50]


plt.figure(figsize=(15, 5))
plt.bar(ig50.index, ig50["IG"], color = 'red', label = "Words")
plt.xticks(rotation = 90)
plt.xlabel("Words")
plt.ylabel("Information Gain")
plt.title("Top 50 Information Gain")
plt.show()



ig.sort_values(by = "IG", ascending=False)[:50]
