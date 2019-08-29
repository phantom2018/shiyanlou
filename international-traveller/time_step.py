import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

seed = 7
batch_size =1
epochs = 100
filename = 'international-airline-passengers.csv'
footer = 3
look_back = 3

def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        x = dataset[i: i+look_back, 0]
        dataX.append(x)
        y = dataset[i+look_back, 0]
        dataY.append(y)
        print('X: %s, y: %s' % (x, y))
    return np.array(dataX), np.array(dataY)


def build_model():
    model  = Sequential()
    model.add(LSTM(units=4, input_shape=(look_back, 1)))
    model.add(Dense(units=1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


if __name__ == '__main__':
    
    np.random.seed(seed)
    
    data=read_csv(filename, usecols=[1], engine='python', skipfooter=footer)
    dataset = data.values.astype('float32')
    
    print('dataset:')
    print(dataset)
    
    
    #?????
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset)*0.67)
    validation_size = len(dataset) - train_size
    train, validation = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print('train:')
    print(train)
    print('validation:')
    print(validation)
    
    #??dataset, ?????????
    X_train, y_train = create_dataset(train)
    X_validation, y_validation = create_dataset(validation)
    print('X_train:')
    print(X_train)
    print('y_train:')
    print(y_train)
    
    
    #?????????? ????? ???
    X_train = np.reshape(X_train, (X_train.shape[0],  X_train.shape[1], 1))
    X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))
    print('X_train2:')
    print(X_train)
    print('X_validation2:')
    print(X_validation)
    
    model = build_model()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)
    
    predict_train = model.predict(X_train)
    print('predicted train:')
    print(predict_train)
    predict_validation = model.predict(X_validation)
    print('predicted vali:')
    print(predict_validation)
    
    #??????? ?????MSE????
    predict_train = scaler.inverse_transform(predict_train)
    print('predicted train:')
    print(predict_train)
    y_train = scaler.inverse_transform([y_train])
    print('y_train:')
    print(y_train)
    predict_validation = scaler.inverse_transform(predict_validation)
    print('predicted vali:')
    print(predict_validation)
    y_validation = scaler.inverse_transform([y_validation])
    print('y vali:')
    print(y_validation)
    
    
    train_score = math.sqrt(mean_squared_error(y_train[0], predict_train[:, 0]))
    print('Train score: %.2f RMSE' % (train_score))
    validation_score = math.sqrt(mean_squared_error(y_validation[0], predict_validation[:, 0]))
    print('Validation score: %.2f RMSE' % (validation_score))
    
    #?? ???????????? ????
    predict_train_plot = np.empty_like(dataset)
    predict_train_plot[:, :] = np.nan
    predict_train_plot[look_back:len(predict_train) + look_back, :] = predict_train
    
    #?? ????????????????
    predict_validation_plot = np.empty_like(dataset)
    predict_validation_plot[:, :] = np.nan
    predict_validation_plot[len(predict_train) + look_back*2+1 : len(dataset)-1 :] = predict_validation
    
    #??
    dataset = scaler.inverse_transform(dataset)
    plt.plot(dataset, color='blue')
    plt.plot(predict_train_plot, color='green')
    plt.plot(predict_validation_plot, color='red')
    plt.show()

