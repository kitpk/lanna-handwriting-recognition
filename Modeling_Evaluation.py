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

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D, Softmax, AvgPool2D
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical, plot_model

import time
import json
import ast

with open("config.json") as json_data_file:
    data = json.load(json_data_file)

# image_size = data["data_preprocessing"]["image_size"]
# class_names = data["modeling"]["class_names"]
# col = data["modeling"]["histogram"][0]["col"]
# row = data["modeling"]["histogram"][0]["row"]

# dataset_name = "windows_test.csv"


from Data_Splitting import split_train_test

# X_train, X_test, y_train, y_test = split_train_test(
#     dataset_name, image_size, class_names
# )


# Histogram Modeling and Evaluation
def Histogram(X_train, X_test, y_train, y_test):
    data1 = []
    for flatten in X_train.values:
        image = np.reshape(flatten, (image_size, image_size, 1))
        data1.append(image)
    X_train_t = np.array(data1, dtype=np.float32)

    data2 = []
    for flatten in X_test.values:
        image = np.reshape(flatten, (image_size, image_size, 1))
        data2.append(image)
    X_test_t = np.array(data2, dtype=np.float32)

    Y_train = to_categorical(y_train)
    Y_test = to_categorical(y_test)

    X_train_t = X_train_t / 255.0
    X_test_t = X_test_t / 255.0

    # Start Training model
    print("START Training")
    start_time = time.time()
    identity = create_identity(X_train_t, Y_train)
    # Save model
    filename = f"MODEL/{dataset_name}_{col}x{row}_Histogram.csv"
    df = pd.DataFrame(identity)
    df.to_csv(filename, index=False)
    # End Training model
    end_time = time.time()
    print("End Training")
    print("Function execution time:", end_time - start_time, "seconds")

    # Start Evaluation
    print("START Evaluation")
    print("Filename =", filename)
    model_HG = pd.read_csv(filename)
    target_predicted = model_predict(X_test_t, model_HG)
    target_Y_test = np.argmax(Y_test, axis=1)
    print("Predicted:", target_predicted)
    # Confusion Matrix
    cm = metrics.confusion_matrix(target_Y_test, target_predicted)
    print("Confusion Matrix:", cm)
    # precision, Recall, F1-Score
    print("\nREPORT")
    print(metrics.classification_report(target_Y_test, target_predicted))
    print(
        "\nMetrics SCORE Y_test:",
        metrics.accuracy_score(target_Y_test, target_predicted),
        "--> %.2f%%" % (metrics.accuracy_score(target_Y_test, target_predicted) * 100),
    )
    # RMSE
    length = len(target_Y_test)
    y = np.array([1 for i in range(length)])
    yhat = np.array(
        [1 if target_Y_test[i] == target_predicted[i] else 0 for i in range(length)]
    )
    d = y - yhat
    mse_f = np.mean(d**2)
    rmse_f = np.sqrt(mse_f)
    print("RMSE SCORE Y_test:", rmse_f, "--> %.2f%%" % (rmse_f * 100))
    # End Evaluation


# For Histogram Modeling
def create_identity(X_train_t, Y_train):
    # Create identity
    c_identity = divide_grid_line(X_train_t, Y_train, col, row)
    # Format c_identity
    len_max = check_len_data(c_identity)
    add_none(c_identity, len_max)
    return c_identity


def divide_grid_line(images, label, col, row):
    identity = {_: [] for _ in range(42)}
    for num in range(len(images)):
        grid_col = images[num].shape[0] // col
        grid_row = images[num].shape[1] // row

        line_col = [0]
        line_row = [0]

        data_many = []
        c = 0
        for i in range(1, col + 1):
            x_col = i * grid_col
            line_col.append(x_col)
            for j in range(1, row + 1):
                if i == 1:
                    x_row = j * grid_row
                    line_row.append(x_row)
                pixel_white = cv2.countNonZero(
                    images[num][
                        line_row[j - 1] : line_row[j], line_col[i - 1] : line_col[i]
                    ]
                )
                size = images[num][
                    line_row[j - 1] : line_row[j], line_col[i - 1] : line_col[i]
                ].size
                pixel_black = size - pixel_white
                data_many.append(pixel_black)
                c += 1
        identity[np.argmax(label[num])].append(data_many)

    return identity


def check_len_data(identity):
    len_max = []
    for i in identity:
        len_max.append(len(identity[i]))
        # print(len(identity[i]))
    return max(len_max)


def add_none(identity, len_max):
    for i in identity:
        if len(identity[i]) != len_max:
            for j in range(0, len_max - len(identity[i])):
                identity[i].append(None)


# For Histogram Evaluation
def model_predict(image, model):
    target_predicted = []
    for i in range(len(image)):
        c_identity = divide_grid_line_evaluation(image[i], col, row)
        accuracy = calculate_matching(c_identity, model)
        # print("accuracy", accuracy)
        # print("image ", i)
        ac_max = max(accuracy.values())
        predicted = search(ac_max, accuracy)
        target_predicted.append(predicted)
    # print("target_predicted", target_predicted)
    return np.array(target_predicted)


def divide_grid_line_evaluation(image, col, row):
    grid_col = image.shape[0] // (col)
    grid_row = image.shape[1] // (row)
    line_col = [0]
    line_row = [0]
    data_one = []
    c = 0
    for i in range(1, col + 1):
        x_col = i * grid_col
        line_col.append(x_col)
        for j in range(1, row + 1):
            if i == 1:
                x_row = j * grid_row
                line_row.append(x_row)
            pixel_white = cv2.countNonZero(
                image[line_row[j - 1] : line_row[j], line_col[i - 1] : line_col[i]]
            )
            size = image[
                line_row[j - 1] : line_row[j], line_col[i - 1] : line_col[i]
            ].size
            pixel_black = size - pixel_white
            data_one.append(pixel_black)
            c += 1
    return data_one


def calculate_matching(histogram, model):
    accuracy = {_: [] for _ in range(42)}
    for i in range(model.shape[1]):
        MODEL_data = [model[f"{i}"][_] for _ in range(model.shape[0])]
        for j in range(model.shape[0]):
            if type(MODEL_data[j]) == str:
                data_list = ast.literal_eval(MODEL_data[j])
                T = data_list
                S = histogram
                n = len(T)
                sum = 0
                for k in range(n):
                    if T[k] == 0:
                        T[k] = 1
                        S[k] += 1
                        sum = condition_cal(sum, T, S, k)
                        S[k] -= 1
                    else:
                        sum = condition_cal(sum, T, S, k)
                cal = 100 - (sum / n) * 100
                accuracy[i].append(cal)
    for z in accuracy:
        accuracy[z] = max(accuracy[z])
    return accuracy


def condition_cal(sum, T, S, index):
    if T[index] < S[index]:
        if S[index] / T[index] > 2:
            sum += 1
        else:
            sum += ((S[index] - T[index])) / T[index]
    elif T[index] == S[index]:
        sum += 0
    else:
        sum += (T[index] - S[index]) / T[index]
    return sum


def search(search_value, accuracy):
    for key, value in accuracy.items():
        if value == search_value:
            return key


# Convolutional Neural Network Modeling and Evaluation
def Convolutional_Neural_Network(X_train, X_test, y_train, y_test):
    data1 = []
    for flatten in X_train.values:
        image = np.reshape(flatten, (image_size, image_size, 1))
        data1.append(image)
    X_train_t = np.array(data1, dtype=np.float32)

    data2 = []
    for flatten in X_test.values:
        image = np.reshape(flatten, (image_size, image_size, 1))
        data2.append(image)
    X_test_t = np.array(data2, dtype=np.float32)

    Y_train = to_categorical(y_train)
    Y_test = to_categorical(y_test)

    X_train_t = X_train_t / 255.0
    X_test_t = X_test_t / 255.0

    # Set parameters = Input, Feature Extraction, Classification
    model = Sequential(
        [
            Conv2D(6, (5, 5), activation="relu", input_shape=(image_size, image_size, 1)),
            AvgPool2D((2, 2), 2),
            Conv2D(16, (5, 5), activation="relu"),
            AvgPool2D((2, 2), 2),
            Flatten(),
            Dense(84, activation="relu"),
            Dense(42, activation="softmax"),  # softmax for one hot . . # sigmoid for 0/1
        ]
    )

    # Set parameters = Output
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    best_loss_checkpoint = ModelCheckpoint(
        filepath="../MODEL/train/best_loss_model.h5",
        monitor="loss",
        save_best_only=True,
        save_weights_only=True,
        node="min",
    )

    best_val_loss_checkpoint = ModelCheckpoint(
        filepath="../MODEL/train/best_val_loss_model.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        node="min",
    )

    # Start Training model
    print("START Training")
    start_time = time.time()
    history = model.fit(
        X_train_t, Y_train, batch_size=32, epochs=100, validation_data=(X_test_t, Y_test)
    )
    # Save model
    filename = f"MODEL/{dataset_name}_CNN.h5"
    # NOTE model.save(filename)
    # End Training model
    end_time = time.time()
    print("End Training")
    print("Function execution time:", end_time - start_time, "seconds")

    # Start Evaluation
    print("START Evaluation")
    print("Filename =", filename)
    predicted = model.predict(X_test_t)
    target_predicted = np.argmax(predicted, axis=1)
    target_Y_test = np.argmax(Y_test, axis=1)
    print("Predicted:", target_predicted)
    # Confusion Matrix
    cm = metrics.confusion_matrix(target_Y_test, target_predicted)
    print("Confusion Matrix:", cm)
    # precision, Recall, F1-Score
    print("\nREPORT")
    print(metrics.classification_report(target_Y_test, target_predicted))
    print(
        "\nMetrics SCORE Y_test:",
        metrics.accuracy_score(target_Y_test, target_predicted),
        "--> %.2f%%" % (metrics.accuracy_score(target_Y_test, target_predicted) * 100),
    )
    # RMSE
    length = len(target_Y_test)
    y = np.array([1 for i in range(length)])
    yhat = np.array(
        [1 if target_Y_test[i] == target_predicted[i] else 0 for i in range(length)]
    )
    d = y - yhat
    mse_f = np.mean(d**2)
    rmse_f = np.sqrt(mse_f)
    print("RMSE SCORE Y_test:", rmse_f, "--> %.2f%%" % (rmse_f * 100))
    # End Evaluation


# Logistic Regression Modeling and Evaluation
def Logistic_Regression(X_train, X_test, y_train, y_test):
    data1 = []
    for flatten in X_train.values:
        image = np.reshape(flatten, (image_size * image_size))
        data1.append(image)
    X_train_t = np.array(data1, dtype=np.float32)

    data2 = []
    for flatten in X_test.values:
        image = np.reshape(flatten, (image_size * image_size))
        data2.append(image)
    X_test_t = np.array(data2, dtype=np.float32)

    X_train_t = X_train_t / 255.0
    X_test_t = X_test_t / 255.0

    model = LogisticRegression(multi_class="auto")

    # Start Training model
    print("START Training")
    start_time = time.time()
    model.fit(X_train_t, y_train)
    # Save model
    filename = f"MODEL/{dataset_name}_LOG.h5"
    pickle.dump(model, open(filename, "wb"))
    # End Training model
    end_time = time.time()
    print("End Training")
    print("Function execution time:", end_time - start_time, "seconds")

    # Start Evaluation
    print("START Evaluation")
    print("Filename =", filename)
    target_predicted = model.predict(X_test_t)
    target_Y_test = np.argmax(to_categorical(y_test), axis=1)
    print("Predicted:", target_predicted)
    # Confusion Matrix
    cm = metrics.confusion_matrix(target_Y_test, target_predicted)
    print("Confusion Matrix:", cm)
    # precision, Recall, F1-Score
    print("\nREPORT")
    print(metrics.classification_report(target_Y_test, target_predicted))
    print(
        "\nMetrics SCORE Y_test:",
        metrics.accuracy_score(target_Y_test, target_predicted),
        "--> %.2f%%" % (metrics.accuracy_score(target_Y_test, target_predicted) * 100),
    )
    # RMSE
    length = len(target_Y_test)
    y = np.array([1 for i in range(length)])
    yhat = np.array(
        [1 if target_Y_test[i] == target_predicted[i] else 0 for i in range(length)]
    )
    d = y - yhat
    mse_f = np.mean(d**2)
    rmse_f = np.sqrt(mse_f)
    print("RMSE SCORE Y_test:", rmse_f, "--> %.2f%%" % (rmse_f * 100))
    # End Evaluation


def Vote_Ensemble(X_test, y_test):
    data2 = []
    for flatten in X_test.values:
        image = np.reshape(flatten, (image_size, image_size, 1))
        data2.append(image)
    X_test_t = np.array(data2, dtype=np.float32)

    Y_test = to_categorical(y_test)

    X_test_t = X_test_t / 255.0

    # Start Evaluation
    print("START Evaluation")
    # Histogram
    filename = f"MODEL/{model_name}_{col}x{row}_Histogram.csv"
    print("Filename =", filename)
    model_HG = pd.read_csv(filename)
    # CNN
    filename = f"MODEL/{model_name}_CNN.h5"
    print("Filename =", filename)
    model_CNN = load_model(filename)
    # LOG
    filename = f"MODEL/{model_name}_LOG.h5"
    print("Filename =", filename)
    model_LOG = pickle.load(open(filename, "rb"))
    # Vote Ensemble
    target_predicted = model_predict_vote_ensemble(
        X_test_t, model_HG, model_CNN, model_LOG
    )
    target_Y_test = np.argmax(Y_test, axis=1)
    print("Predicted:", target_predicted)
    # Confusion Matrix
    cm = metrics.confusion_matrix(target_Y_test, target_predicted)
    print("Confusion Matrix:", cm)
    # precision, Recall, F1-Score
    print("\nREPORT")
    print(metrics.classification_report(target_Y_test, target_predicted))
    print(
        "\nMetrics SCORE Y_test:",
        metrics.accuracy_score(target_Y_test, target_predicted),
        "--> %.2f%%" % (metrics.accuracy_score(target_Y_test, target_predicted) * 100),
    )
    # RMSE
    length = len(target_Y_test)
    y = np.array([1 for i in range(length)])
    yhat = np.array(
        [1 if target_Y_test[i] == target_predicted[i] else 0 for i in range(length)]
    )
    d = y - yhat
    mse_f = np.mean(d**2)
    rmse_f = np.sqrt(mse_f)
    print("RMSE SCORE Y_test:", rmse_f, "--> %.2f%%" % (rmse_f * 100))
    # End Evaluation


# For Vote Ensemble Evaluation
def model_predict_vote_ensemble(image, model_HG, model_CNN, model_LOG):
    target_predicted = []
    print("image = ", image.shape)
    print("image[0] = ", image[0].shape)

    for i in range(len(image)):
        # predicted
        predict_CNN = model_CNN.predict(np.reshape(image[i], (1, 64, 64, 1)))
        predict_LOG = model_LOG.predict(np.reshape(image[i], (1, 64*64)))

        if np.argmax(predict_CNN) == np.argmax(predict_LOG):
            target_predicted.append(np.argmax(predict_CNN[0]))
        else:
            c_identity = divide_grid_line_evaluation(image[i], col, row)
            accuracy = calculate_matching(c_identity, model_HG)
            # print("accuracy", accuracy)
            # print("image ", i)
            CNN = accuracy[np.argmax(predict_CNN[0])]
            LOG = accuracy[predict_LOG[0]]
            if CNN >= LOG:
                target_predicted.append(np.argmax(predict_CNN[0]))
            else:
                target_predicted.append(predict_LOG[0])

    return np.array(target_predicted)


if __name__ == "__main__":
    image_size = data["data_preprocessing"]["image_size"]
    class_names = data["modeling"]["class_names"]

    model_name = "windows_train_test.csv"
    dataset_name = "windows_train_test.csv"

    X_train, X_test, y_train, y_test = split_train_test(dataset_name)

    col = data["modeling"]["histogram"][0]["col"]
    row = data["modeling"]["histogram"][0]["row"]

    print("X_train",X_train.shape)
    print("X_train",X_test.shape)
    
    # for data in data["modeling"]["histogram"]:
    #     col = data["col"]
    #     row = data["row"]
    #     print("START Modeling")
    #     Histogram(X_train, X_test, y_train, y_test)

    Convolutional_Neural_Network(X_train, X_test, y_train, y_test)
    # Logistic_Regression(X_train, X_test, y_train, y_test)

    # for data in data["modeling"]["vote_ensemble"]:
    #     col = data["col"]
    #     row = data["row"]
    #     model_name = "windows_train_test.csv"
    #     dataset_name = "windows_experiment"
    #     Vote_Ensemble(X_test, y_test)
    # print("END Modeling")
