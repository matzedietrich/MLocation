# trying to set the python hash seed from script (not working on windows)
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] ='0'

#set random seed for random and numpy.random
import numpy as np
import random as rn
rn.seed(1)
np.random.seed(1)

# import tensorflow set random seed for tensorflow
import tensorflow as tf
graph_level_seed = 1
operation_level_seed = 2
tf.random.set_seed(graph_level_seed)

# force use of CPU
from tensorflow.compat.v1.keras import backend as k
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
allow_soft_placement=True, device_count = {'CPU': 1})
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=config)
k.set_session(sess)

# force use of one thread only
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# force manual variable initialization
from tensorflow.compat.v1.keras.backend import manual_variable_initialization
manual_variable_initialization(True)


#importing pandas & tensorflow functions
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dense, Activation, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2


# configuration variables
maxlen = 40
labels = 12
vocab = []
len_vocab = 0
char_index = []


# read data
col_list = ["name", "state"]
df = pd.read_csv('data/geographic_names_dataset.csv', sep=";", usecols=col_list)

names = df['name'].apply(lambda x: str(x).lower())
states = df['state']


vocab = set(' '.join([str(i) for i in names]))
vocab.add('END')
len_vocab = len(vocab)
char_index = dict((c, i) for i, c in enumerate(vocab))

print(char_index)


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


X = []
y = []


X = prepare_X(names)
y = prepare_y(states)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)



model = Sequential()
model.add(Bidirectional(LSTM(512, return_sequences=True), backward_layer=LSTM(512, return_sequences=True, go_backwards=True), input_shape=(maxlen, len_vocab)))
model.add(Dropout(0.3, seed=operation_level_seed))
model.add(Bidirectional(LSTM(512)))
model.add(Dropout(0.3, seed=operation_level_seed))
model.add(Dense(12, activity_regularizer=l2(0.002)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

callback = EarlyStopping(monitor='val_loss', patience=5)
mc = ModelCheckpoint('test_model2', monitor='val_loss', mode='min', verbose=1)
reduce_lr_acc = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, verbose=1, min_delta=1e-4, mode='max')


batch_size = 128
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=15, verbose=1, validation_data=(X_test, y_test), callbacks=[callback, mc, reduce_lr_acc], shuffle=False)

#saver = tf.compat.v1.train.Saver() 
#sess = tf.compat.v1.keras.backend.get_session() 
#saver.save(sess, 'keras_session/session.ckpt') 

