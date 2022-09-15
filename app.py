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

def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])
    # Centralize and crop
    crop_img = img[int(h/2-(min_size/2)):int(h/2+(min_size/2)), int(w/2-(min_size/2)):int(w/2+(min_size/2))]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized

@app.route('/prediction/<filename>') 
def prediction(filename):
    number = re.search(r"\d",filename)
    actual = number.group()
    mypath = os.path.join('uploads', filename)
    #read image as grayscale
    img_gray = cv2.imread(mypath, cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.resize(img_gray, (80,80), interpolation = cv2.INTER_NEAREST)
    # define a threshold, 128 is the middle of black and white in grayscale
    thresh = 128
    # threshold the image
    img_bin = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    img_bin = np.invert(img_bin)
    img_bin = img_bin.astype('float32')
    crop_rows = img_bin[~np.all(img_bin==0, axis=1), :]
    cropped_image = crop_rows[:, ~np.all(crop_rows==0, axis=0)]
    top, bottom, left, right = [10]*4
    img = cv2.copyMakeBorder(cropped_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    img = crop_square(img, 64, cv2.INTER_AREA)
    img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    img_re = img.reshape(64,64,1)
    img_re /= 255
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
