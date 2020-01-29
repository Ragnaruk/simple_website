"""
https://medium.com/@randerson112358/stock-price-prediction-using-python-machine-learning-e82a039ac2bb
"""
import math
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import datetime

from database import get_price

plt.style.use('fivethirtyeight')


def train_nn(data: list, labels: list):
    plt.figure(figsize=(16, 8))
    plt.title('Price History')
    plt.plot(labels, data)
    plt.xlabel('Date and Time', fontsize=18)
    plt.ylabel('Price USD ($)', fontsize=18)
    plt.show()

    # Converting the array to a numpy array
    dataset = np.array(data).reshape(-1, 1)
    # print("dataset: ", dataset)

    # Get /Compute the number of rows to train the model on
    training_data_len = math.ceil(len(dataset) * .8)
    # print("training_data_len: ", training_data_len)

    # Scale the all of the data to be values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # print("scaled_data: ", scaled_data)

    # Create the scaled training data set
    train_data = scaled_data[0:training_data_len, :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # print("x_train: ", x_train)
    # print("y_train: ", y_train)

    # Reshape the data into the shape accepted by the LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model to have two LSTM layers with 50 neurons and two Dense layers,
    # one with 25 neurons and the other with 1 neuron.
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Save the trained model
    filename = "keras_model.h5"
    model.save(filename)

    print("Finished training neural network and saved it with the name: ", filename)

    return training_data_len, filename


def test_nn(data: list, labels: list, training_data_len: int, filename: str):
    model = load_model(filename)

    # Converting the array to a numpy array
    dataset = np.array(data).reshape(-1, 1)
    # print("dataset: ", dataset)

    # Scale the all of the data to be values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # print("scaled_data: ", scaled_data)

    # Test data set
    test_data = scaled_data[training_data_len - 60:, :]

    # Create the x_test and y_test data sets
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert x_test to a numpy array
    x_test = np.array(x_test)

    # Reshape the data into the shape accepted by the LSTM
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    print("x_test: ", x_test)

    # Getting the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)  # Undo scaling

    # Calculate/Get the value of RMSE
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print("rmse: ", rmse)

    # Plot/Create the data for the graph
    train = data[:training_data_len]
    valid = data[training_data_len:]

    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Price USD ($)', fontsize=18)
    plt.plot(labels[:training_data_len], train)
    plt.plot(labels[training_data_len:], valid)
    plt.plot(labels[training_data_len:], predictions)
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()


def get_prediction(data: list):
    model = load_model("keras_model.h5")

    dataset = np.array(data).reshape(-1, 1)
    # print("dataset: ", dataset)

    # Scale the all of the data to be values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # print("scaled_data: ", scaled_data)

    # Test data set
    test_data = scaled_data[training_data_len - 60:, :]

    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    pred_price = model.predict(x_test)
    pred_price = scaler.inverse_transform(pred_price)

    return pred_price


if __name__ == '__main__':
    # Timestamps from current day to the first
    # 1580292247000
    # 1580205847000
    # 1580119447000
    # 1580033047000
    # 1579946647000
    # 1579860247000
    # 1579773847000
    # 1579687447000
    # 1579601047000
    # 1579514647000
    # 1579428247000
    # 1579341847000
    # 1579255447000
    # 1579169047000
    # 1579082647000
    # 1578996247000
    # Training data ahead
    # 1578909847000
    # 1578823447000
    # 1578737047000
    # 1578650647000
    # 1578564247000
    # 1578477847000
    # 1578391447000
    # 1578305047000
    # 1578218647000
    # 1578132247000
    # 1578045847000
    # 1577959447000
    # 1577873047000
    # 1577786647000
    # Day: 86400000
    train = False

    # Training nn
    if train:
        price = get_price("bitcoin", "1577786647000", "1579687447000")
        labels = [datetime.utcfromtimestamp(int(p[0]) / 1000) for p in price]
        data = [p[1] for p in price]

        training_data_len, filename = train_nn(data, labels)

        with open('data/nn_info.pickle', 'wb') as file:
            pickle.dump((training_data_len, filename), file)

    with open('data/nn_info.pickle', 'rb') as file:
        training_data_len, filename = pickle.load(file)

    # Testing nn
    price = get_price("bitcoin", "1577786647000", "1580292247000")
    labels = [datetime.utcfromtimestamp(int(p[0]) / 1000) for p in price]
    data = [p[1] for p in price]

    test_nn(data, labels, training_data_len, filename)
    # print(get_prediction(data))
