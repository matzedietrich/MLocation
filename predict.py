import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
import os
import prepareData


maxlen = 40
labels = 12
vocab = []
len_vocab = 0
char_index = []


col_list = ["name", "state"]
df = pd.read_csv('data/geographic_names_dataset.csv', sep=";", usecols=col_list)

names = df['name']
states = df['state']


vocab = set(' '.join([str(i) for i in names]))
vocab.add('END')
len_vocab = len(vocab)
char_index = dict((c, i) for i, c in enumerate(vocab))


def set_flag(i):
    tmp = np.zeros(len_vocab)
    tmp[i] = 1
    return list(tmp)


def prepare_X(X):
    new_list = []
    trunc_train_name = [str(i)[0:maxlen] for i in X]

    for i in trunc_train_name:
        tmp = [set_flag(char_index[j]) for j in str(i)]
        for k in range(0, maxlen - len(str(i))):
            tmp.append(set_flag(char_index["END"]))
        new_list.append(tmp)

    return new_list


def prepare_y(y):
    new_list = []
    for i in y:
        if i == 'Schleswig-Holstein':
            new_list.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif i == 'Niedersachsen':
            new_list.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif i == 'Nordrhein-Westfalen':
            new_list.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif i == 'Hessen':
            new_list.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif i == 'Rheinland-Pfalz':
            new_list.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif i == 'Baden-Württemberg':
            new_list.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif i == 'Bayern':
            new_list.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif i == 'Brandenburg':
            new_list.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif i == 'Mecklenburg-Vorpommern':
            new_list.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif i == 'Sachsen':
            new_list.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif i == 'Sachsen-Anhalt':
            new_list.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif i == 'Thüringen':
            new_list.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    return new_list


X_pred = []


model = tf.keras.models.load_model('best_model_9.h5')


new_names = ["Marburg"]
X_pred = prepare_X(new_names)
prediction = model.predict(X_pred)


dict_answer = ['Schleswig-Holstein', 'Niedersachsen', 'Nordrhein-Westfalen', 'Hessen', 'Rheinland-Pfalz', 'Baden-Württemberg', 'Bayern', 'Brandenburg', 'Mecklenburg-Vorpommern', 'Sachsen', 'Sachsen-Anhalt', 'Thüringen']


def pred(new_names, prediction, dict_answer):
    return_results = []
    k = 0
    for i in prediction:
        print(i)
        return_results.append([new_names[k], dict_answer[np.argmax(i)]])
        k += 1
    return return_results


print(pred(new_names, prediction, dict_answer))



