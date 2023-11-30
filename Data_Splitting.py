import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import cv2

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D, Softmax
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical, plot_model

import json

with open("config.json") as json_data_file:
    data = json.load(json_data_file)


def split_train_test(dataset_name):
    # read CSV data
    dataset_name = "windows_train_test.csv"
    df = pd.read_csv("DATASET/" + dataset_name)

    # train test split
    X = df[df.columns[df.columns.str.startswith("pixel")]]
    Y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=5, stratify=Y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    dataset_name = "windows_test.csv"
    print("START Data_Splittig")
    X_train, X_test, y_train, y_test = split_train_test(dataset_name)
    print("END Data_Splittig")
