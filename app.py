import os 
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np
#import math
import re

from keras.models import load_model 
from keras.backend import set_session
from skimage.transform import resize 

print("Loading model") 
global model 
model = load_model('handwriting.h5') 

@app.route('/', methods=['GET', 'POST']) 
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')

@app.route('/prediction/<filename>') 
def prediction(filename):
    number = re.search(r"\d",filename)
    actual = number.group()
    my_img = plt.imread(os.path.join('uploads', filename))
    img = resize(my_img, (128, 128, 1))
    model.run_eagerly=True
    probabilities = model.predict(np.array( [img,] ))[0,:]
    print(probabilities)
    number_to_class = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    index = np.argsort(probabilities)
    #marks = math.trunc(probabilities[index[9]] * 100)
    if number_to_class[index[9]] == actual:
     grade = "Good Job!"
    else:
     grade = "Hmmm ... Did I make a wrong guess?"
    predictions = {
      "actual":actual,
      "digit":number_to_class[index[9]],
      "prob" :probabilities[index[9]],
      "comment":grade
     }
    return render_template('predict.html', predictions=predictions)

@app.route('/testimonials/', methods=['GET']) 
def testimonials():
    return render_template('testimonials.html')

@app.route('/about/', methods=['GET']) 
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
