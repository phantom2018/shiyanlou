
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

seed=7
np.random.seed(seed)
# 导入数据
dataset = datasets.load_iris()

x = dataset.data[0:140, :]
Y = dataset.target[0:140]
Y_labels = to_categorical(Y, num_classes=3)

# 构建模型函数
def create_model(optimizer='adam', init='glorot_uniform'):
    # 构建模型
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = create_model()

filepath = 'weights.besty.h5'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callback_list = [checkpoint]

model.fit(x, Y_labels, validation_split=0.2,  epochs=200, batch_size=5, verbose=0, callbacks=callback_list)

i = dataset.data[140:150, :]
J = model.predict(x=i, batch_size=5, verbose=0)
print(J)

