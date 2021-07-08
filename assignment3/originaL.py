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
from math import log2

review = pd.read_csv("../final/review100.csv")

def clean_row(row):
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
review['movie review'] = review['movie review'].apply(clean_row)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cv_matrix = cv.fit_transform(review["movie review"].values)

vocab = cv.get_feature_names()

df_bow = pd.DataFrame(cv_matrix.toarray(), columns=vocab)
most_common = df_bow.sum().sort_values(ascending = False)[:1000]
most_common
df_1000 = pd.DataFrame(np.where(df_bow[most_common.index]>0, 1, 0), columns = most_common.index)
df_1000["sentiment"] = review["sentiment"]


def entropy(df, column):
    aggr = 0
    for i in df[column].unique():
        aggr += -(df[column].value_counts()[i] / len(df)) * log2(df[column].value_counts()[i] / len(df))

    return aggr

def conditional_entropy(df, column, syscolumn):
        aggr = 0
        entropies = {}
        for i in df[column].unique():
            aggr += (df[column].value_counts()[i] / len(df)) * entropy(df[df[column] == i], syscolumn)
            entropies[i] = [entropy(df[df[column] == i], syscolumn)]
        return [aggr, entropies]
def information_gain(df, column, syscolumn):

    return [entropy(df, syscolumn) - conditional_entropy(df, column, syscolumn)[0], conditional_entropy(df, column, syscolumn)[0]]


ig = pd.DataFrame([information_gain(df_1000, i, 'sentiment') for i in df_1000.columns[:-1]], columns = ['IG', "CE"], index =df_1000.columns[:-1])
print(ig.sort_values(by = "IG", ascending=False)[:50])
