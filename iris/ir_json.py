
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

seed=7
np.random.seed(seed)
# 导入数据
dataset = datasets.load_iris()

x = dataset.data[0:140, :]
Y = dataset.target[0:140]
x_train, x_increment, Y_train, Y_increment = train_test_split(x, Y, test_size=0.2, random_state=seed)
Y_train_labels = to_categorical(Y_train, num_classes=3)

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
model.fit(x_train, Y_train_labels, epochs=10, batch_size=5, verbose=2)

i = dataset.data[140:150, :]
J = model.predict(x=i, batch_size=5, verbose=0)
print(J)

scores = model.evaluate(x_train, Y_train_labels, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))

#save model as JSON
model_json = model.to_json()
with open('model.json', 'w') as file:
    file.write(model_json)

#save weights
model.save_weights('model.json.h5')

with open('model.json', 'r') as file:
    model_json = file.read()

#load new model
new_model = model_from_json(model_json)
new_model.load_weights('model.json.h5')

new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

Y_increment_labels = to_categorical(Y_increment, num_classes=3)
new_model.fit(x_increment, Y_increment_labels, epochs=10, batch_size=5, verbose=2)
i = dataset.data[140:150, :]
J = new_model.predict(x=i, batch_size=5, verbose=0)
print(J)

scores = new_model.evaluate(x_increment, Y_increment_labels, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
