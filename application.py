from flask import Flask, render_template, url_for, copy_current_request_context

__author__ = 'matthias dietrich'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True


@app.route('/')
def index():
    # show index.html
    return render_template('index.html')

