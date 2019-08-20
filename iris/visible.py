
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
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

history = model.fit(x, Y_labels, validation_split=0.2,  epochs=200, batch_size=5, verbose=0, callbacks=callback_list)

i = dataset.data[140:150, :]
J = model.predict(x=i, batch_size=5, verbose=0)
print(J)

scores = model.evaluate(x, Y_labels, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
