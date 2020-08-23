# MLocation
This is my first project working with machine learning. It is all about identifying locations based on similiarities in their names in specific regions of germany.


## Requirements & Setup
You need to have [Python 3](https://www.python.org/downloads/), as well as the [pip](https://pypi.org/project/pip/) package manager, installed in order to run the application.

After cloning the repository, navigate into the directory and create a virtual environment:
```python -m venv venv```

Now you need to install all requireded modules via:
```pip install requirements.txt```

It is also very important to set the Python Hash Seed to '0' in order to prevent non-deterministic behavior:
Windows:```$env:PYTHONHASHSEED = '0'```
MacOS, Linux: ```export PYTHONHASHSEED=0```

## Unzip the model
In order to upload my trained model I needed to zip it, to decrease its file size.
To be able to use the trained model simply unzip it.

## (Training the model: optional)
You can also train the model by yourself, although it's not recommended. (It took me 10 hours) Therefore run:
```python train.py```

## Running the application
To run the application use:
```python -m flask run```

Now open your web browser of choice and navigate to:
```http://127.0.0.1:5000/```

You should now see a map of germany next to an input field. Use the input field to type in some location name you have in mind (i.e "Machnow"). Then hit **Predict**.

The map on the left will now show you the probability that the given place name is located in a certain state. The more green a state is, the higher is the propability.


## Evaluate the output
Of course it is not easy to evaluate if a fictional location name would actually be located in a speficic state. The goal of this project was to train a model to find similiarities in the names of locations in different regions of germany.
I found a similiar project [here](http://truth-and-beauty.net/experiments/ach-ingen-zell/) made by Moritz Stefaner. He was looking at specific endings, like "-bach", "-berg", "-dorf" or "-hausen", that appear in some regions more frequently. I used his results to evaluate my trained model. For example,when you are choosing a location name with **-ow** or **-itz** at the end, the model predicts it to be in eastern germany.


## Next steps
I am currently looking for ways to visualize what the model has learned and what patterns it's trying to identify in order to predict the state. A very promising project about LSTM Visualization can be found [here](http://lstm.seas.harvard.edu/).