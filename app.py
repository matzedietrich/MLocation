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
            return render_template('index.html', prediction=prediction)
        return redirect(request.url)

    return render_template('index.html')



