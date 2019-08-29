from pandas import DataFrame, concat, read_csv
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime
from matplotlib import pyplot as plt

batch_size=72
epochs=50
n_input=1
n_train_hours = 365*24*4
n_validation_hours = 24*5
lstm_input_shape = 8

filename = 'pollution_original.csv'

def prase(x):
    return datetime.strptime(x, '%Y %m %d %H')

def load_dataset():
    dataset = read_csv(filename, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=prase)
    dataset.drop('No', axis=1, inplace=True)
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wind_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    dataset['pollution'].fillna(dataset['pollution'].mean(), inplace=True)
    return dataset


def convert_dataset(data, n_input=1, out_index=0, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = [] , []
    #?????t-n, ...t-1)
    for i in range(n_input, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    #????t
    cols.append(df[df.columns[out_index]])
    names += ['result']
    print('cols:')
    print(cols)
    print('names:')
    print(names)
    #????/????
    result = concat(cols, axis=1)
    result.columns = names
    #?????????
    if dropnan:
        result.dropna(inplace=True)
    print('convert_dataset_result:')
    print(result)
    return result
        
        
def class_encode(data, class_indexs):
    encoder = LabelEncoder()
    class_indexs = class_indexs if type(class_indexs) is list else [class_indexs]
    values = DataFrame(data).values
    for index in class_indexs:
        values[:, index] = encoder.fit_transform(values[:, index])
    return DataFrame(values) if type(data) is DataFrame else values



def build_models(lstm_input_shape):
    model = Sequential()
    model.add(LSTM(units=50, input_shape=lstm_input_shape, return_sequences=True))
    model.add(LSTM(units=50, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1))
    
    model.compile(loss='mae', optimizer='adam')
    model.summary()
    return model



if __name__ == '__main__':
    data = load_dataset()
    print(data.head(5))
    
    groups = [0, 1, 2, 3, 4, 5, 6, 7]
    plt.figure()
    i=1
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(data.values[:, group])
        plt.title(data.columns[group], y=0.5, loc='right')
        i += 1
    plt.show()
    
    #????????
    data = class_encode(data, 4)
    print('data:')
    print(data)
    
    #?????? ???5??????????
    dataset = convert_dataset(data, n_input=n_input)
    values = dataset.values.astype('float32')
    print('dataset_values:')
    print(values)
    
    #??????????
    train = values[:n_train_hours, :]
    validation = values[-n_validation_hours:, :]
    print('train:')
    print(train)
    print('train.shape:')
    print(train.shape)
    print('validation:')
    print(validation)
    print('validation.shape;')
    print(validation.shape)
    x_train, y_train = train[:, :-1], train[:, -1]
    x_validation, y_validation = validation[:, :-1], validation[:, -1]
    print('x_train:')
    print(x_train)
    print('x_validation:')
    print(x_validation)
    print('y_train:')
    print(y_train)
    print('y_validation:')
    print(y_validation)
    print(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape)
    
    #?????? ????0-1)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_validation = scaler.fit_transform(x_validation)
    print('x_train:')
    print(x_train)
    print('x_validation:')
    print(x_validation)
    print(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape)
    
    #?????????? ????? ?????
    x_train = x_train.reshape(x_train.shape[0], n_input, x_train.shape[1])
    x_validation = x_validation.reshape(x_validation.shape[0], 1, x_validation.shape[1])
    print('x_train2:')
    print(x_train)
    print('x_validation2:')
    print(x_validation)
    print(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape)
    
    lstm_input_shape = (x_train.shape[1], x_train.shape[2])
    model = build_models(lstm_input_shape)
    model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_validation, y_validation), epochs=epochs, verbose=2)
    
    prediction = model.predict(x_validation)
    
    plt.plot(y_validation, color='blue', label='Actual')
    plt.plot(prediction, color='green', label='Prediction')
    plt.legend(loc='upper right')
    plt.show()
    
    

