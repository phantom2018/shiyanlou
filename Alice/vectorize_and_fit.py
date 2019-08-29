from nltk import word_tokenize
from gensim import corpora
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
import numpy as np
from keras.utils import np_utils
from pyecharts.charts import WordCloud


filename = 'alice.txt'
document_split = ['.', ',', '?', '!', ';']
batch_size = 128
epochs = 100
model_json_file = 'simple_model.json'
model_hd5_file = 'simple_model.hd5'
dict_file = 'dict_file.txt'
dict_len = 2790
max_len = 20
document_max_len = 33200

def load_dataset():
    with open(filename, 'r') as file:
        document = []
        lines = file.readlines()
        for line in lines:
            #???????
            value = clear_data(line)
            if value != '':
                for str in word_tokenize(value):
                    if str == 'CHAPTER':
                        break
                    else:
                        document.append(str.lower())
    return document

def clear_data(str):
    #???????????????
    value = str.replace('\ufeff', '').replace('\n', '')
    return value

def word_to_integer(document):
    #????
    dic = corpora.Dictionary([document])
    print('dic:')
    print(dic)
    #?????????
    dic.save_as_text(dict_file)
    dic_set = dic.token2id
    print('dic_set:')
    print(dic_set)
    #????????
    values = []
    for word in document:
        #?????????????
        values.append(dic_set[word])
    print('values:')
    print(values)                             
    return values



def make_y(document):
    dataset = make_dataset(document)
    y = dataset[1:dataset.shape[0], 0]
    print('y:')
    print(y)
    return y
    
def make_x(document):
    dataset = make_dataset(document)
    x = dataset[0:dataset.shape[0]-1, :]
    print('x:')
    print(x)
    return x
    
    
    
def make_dataset(document):
    dataset = np.array(document[0:document_max_len])
    dataset = dataset.reshape(int(document_max_len / max_len), max_len)
    return dataset

"""
def show_word_cloud(document):
    #?????????
    left_words = ['.', ',', '?', '!',';', ':', '\'', '(', ')']
    #????
    dic = corpora.Dictionary([document])
    #?????????
    words_set = dic.doc2bow(document)
    
    words, frequences = [], []
    for item in words_set:
        key = item[0]
        frequence = item[1]
        word = dic.get(key=key)
        if word not in left_words:
            words.append(word)
            frequences.append(frequence)
    
    #??pyecharts????
    word_cloud = WordCloud()
    word_cloud.add(name='Alice\'s word cloud', attr=words, value=frequences, shape='circle', word_size_range=[20, 100])
    word_cloud.render()
"""
    
def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=dict_len, output_dim=32, input_length=max_len))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=256))
    model.add(Dropout(0.2))
    model.add(Dense(units=dict_len, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model


if __name__ == '__main__':
    print('begin')
    document = load_dataset()
    print('document:')
    print(document)
    #show_word_cloud(document)
    
    #????????
    values = word_to_integer(document)
    x_train = make_x(values)
    y_train = make_y(values)
    #??????
    y_train = np_utils.to_categorical(y_train, dict_len)
    
    model = build_model()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)
    model_json = model.to_json()
    with open(model_json_file, 'w') as file:
        file.write(model_json)
    model.save_weights(model_hd5_file)
print('end')

