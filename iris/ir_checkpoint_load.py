
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

# daoru模型函数
def load_model(optimizer='adam', init='glorot_uniform'):

    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))

    filepath = 'weights.besty.h5'
    model.load_weights(filepath=filepath)
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = load_model()


i = dataset.data[140:150, :]
J = model.predict(x=i, batch_size=5, verbose=0)
print(J)
scores = model.evaluate(x, Y_labels, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
