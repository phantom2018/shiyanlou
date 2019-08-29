from nltk import word_tokenize
from gensim import corpora
from keras.models import model_from_json
import numpy as np

model_json_file = 'simple_model.json'
model_hd5_file = 'simple_model.hd5'
dict_file = 'dict_file.txt'
words = 200
max_len = 20
myfile = 'myfile.txt'


def load_dict():
    dic = corpora.Dictionary.load_from_text(dict_file)
    return dic

def load_model():
    with open(model_json_file, 'r') as file:
        model_json = file.read()
        
    model = model_from_json(model_json)
    model.load_weights(model_hd5_file)
        
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
    
def word_to_integer(document):
    #????
    dic = load_dict()
    print('dic:')
    print(dic)
    dic_set = dic.token2id
    print('dic_set:')
    print(dic_set)
    values = []
    for word in document:
        #?????????????
        values.append(dic_set[word])
    
    return values

def make_dataset(document):
    dataset = np.array(document)
    print('original  x dataset:')
    print(dataset)
    dataset = dataset.reshape(1, max_len)
    print('dataset(x):')
    print(dataset)
    return dataset

def reverse_document(values):
    dic = load_dict()
    dic_set = dic.token2id
    #????????
    document = ''
    for value in values:
        word = dic.get(value)
        document = document + word + ' '
    print('document(word):')
    print(document)
    return document


if __name__ == '__main__':
    model = load_model()
    start_doc = 'Alice is a little girl , \
             who has a dream to go to visit the land in the time.'
    document = word_tokenize(start_doc.lower())
    new_document = []
    values = word_to_integer(document)
    print('values of start_doc:')
    print(values)
    new_document = [] + values
    
    for i in range(words):
        x = make_dataset(values)
        prediction = model.predict(x, verbose=0)
        print('pred:')
        print(prediction)
        prediction = np.argmax(prediction)
        print('pred2:')
        print(prediction)
        values.append(prediction)
        new_document.append(prediction)
        values = values[1: ]
        
    new_document = reverse_document(new_document)
    print('result:')
    print(new_document)
    with open(myfile, 'w') as file:
        file.write(new_document)

