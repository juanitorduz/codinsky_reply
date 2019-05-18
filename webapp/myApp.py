from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import Form
from wtforms import TextField

from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.applications import vgg19
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time
from matplotlib import pyplot as plt

from nltk.sentiment.vader import SentimentIntensityAnalyzer


app = Flask(__name__)
app.config['SECRET_KEY'] = 'our very hard to guess secretfir'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process',methods=['GET', 'POST'])
def process():

    # This is the path to the image you want to transform.
    target_image_path = 'profile_pic.jpg'
    # This is the path to the style image.
    style_reference_image_path = 'kandisky.jpg'
    width, height = load_img(target_image_path).size
    img_height = 400
    img_width = int(width * img_height / height)    
    first_name = request.form['firstname']
    phrase = request.form['textarea']
    print(sentiment_analyzer.polarity_scores(phrase))	
    return render_template('process.html', name=first_name,phrase=phrase)



sentiment_analyzer = SentimentIntensityAnalyzer()	
	
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x	
	
	
	
@app.route('/upload')
def upload():
    return render_template('thank-you.html')	
	
	
# Simple form handling using raw HTML forms
@app.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    error = ""
    if request.method == 'POST':
        # Form being submitted; grab data from form.
        first_name = request.form['textarea']
        print("POST in signup")
        last_name = 'andrea'
        # Validate form data
        if len(first_name) == 0 or len(last_name) == 0:
            # Form data failed validation; try again
            error = "Please supply both first and last name"
        else:
            # Form data is valid; move along
            # return redirect(url_for('thank_you'))
            render_template('thank-you.html')

    # Render the sign-up page
    return render_template('sign-up.html', message=error)

# More powerful approach using WTForms
class RegistrationForm(Form):
    first_name = TextField('First Name')
    last_name = TextField('Last Name')

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = ""
    form = RegistrationForm(request.form)

    if request.method == 'POST':
        first_name = form.first_name.data
        last_name = form.last_name.data

        if len(first_name) == 0 or len(last_name) == 0:
            error = "Please supply both first and last name"
        else:
            return redirect(url_for('thank_you'))

    return render_template('register.html', form=form, message=error)

# Run the application
app.run(debug=True)
