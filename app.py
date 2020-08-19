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
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
allow_soft_placement=True, device_count = {'CPU': 1})
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=config)
k.set_session(sess)

# force use of one thread only
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


from tensorflow.compat.v1.keras.backend import manual_variable_initialization
manual_variable_initialization(True)


from flask import Flask, render_template, url_for, copy_current_request_context
from flask import request, redirect
from predict import predictStateFrom

__author__ = 'matthias dietrich'

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        req = request.form['name']
        prediction = predictStateFrom(req)
        if prediction:
            name = prediction[0].upper()
            sh = prediction[1][0]
            ni = prediction[1][1]
            nw = prediction[1][2]
            he = prediction[1][3]
            rp = prediction[1][4]
            bw = prediction[1][5]
            by = prediction[1][6]
            bb = prediction[1][7]
            mv = prediction[1][8]
            sn = prediction[1][9]
            sa = prediction[1][10]
            th = prediction[1][11]
            return render_template('index.html', name=name, sh=sh, ni=ni, nw=nw, he=he, rp=rp, bw=bw, by=by, bb=bb, mv=mv, sn=sn, sa=sa, th=th)
        return redirect(request.url)

    return render_template('index.html')



