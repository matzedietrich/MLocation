import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] ='0'

import numpy as np
import random as rn
rn.seed(1)
np.random.seed(1)

# set random seed for tensorflow
import tensorflow as tf
graph_level_seed = 1
operation_level_seed = 2
tf.random.set_seed(graph_level_seed)


from tensorflow.compat.v1.keras import backend as k
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True, device_count = {'CPU': 1})
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=config)
k.set_session(sess)

# force use of one thread only
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


from tensorflow.compat.v1.keras.backend import manual_variable_initialization
manual_variable_initialization(True)


import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dense, Activation, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2


maxlen = 40
labels = 12
vocab = []
len_vocab = 0
char_index = []


col_list = ["name", "state"]
df = pd.read_csv('data/geographic_names_dataset.csv', sep=";", usecols=col_list)

names = df['name'].apply(lambda x: str(x).lower())
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


def pred(new_names, prediction, dict_answer):
    return_results = []
    k = 0
    for i in prediction:
        print(i)
        #return_results.append([new_names[k], dict_answer[np.argmax(i)]])
        return_results.append(new_names[k])
        return_results.append(i)
        k += 1
    return return_results


X_pred = []

model = tf.keras.models.load_model('test_model2')

#saver = tf.compat.v1.train.Saver() 
#sess = tf.compat.v1.keras.backend.get_session() 
#saver.restore(sess,'keras_session/session.ckpt') 


dict_answer = ['Schleswig-Holstein', 'Niedersachsen', 'Nordrhein-Westfalen', 'Hessen', 'Rheinland-Pfalz', 'Baden-Württemberg', 'Bayern', 'Brandenburg', 'Mecklenburg-Vorpommern', 'Sachsen', 'Sachsen-Anhalt', 'Thüringen']


def predictStateFrom(input):

    new_names = [input.lower()]
    X_pred = prepare_X(new_names)

    with tf.device("/CPU:0"):
        prediction = model.predict(X_pred)
        return pred(new_names, prediction, dict_answer)
