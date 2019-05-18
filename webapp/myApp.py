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

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

img_width = 200
img_height = 400
sentiment_analyzer = SentimentIntensityAnalyzer()
app = Flask(__name__)
app.config['SECRET_KEY'] = 'our very hard to guess secretfir'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process',methods=['GET', 'POST'])
def process():

    # This is the path to the image you want to transform.
    target_image_path = 'static/91881.jpg'
    # This is the path to the style image.
    style_reference_image_path = 'static/91880.jpg'
    width, height = load_img(target_image_path).size
    img_height = 400
    img_width = int(width * img_height / height)    
    first_name = request.form['firstname']
    phrase = request.form['textarea']
    
    print('Model loaded.')	
    output_dict = get_sentiment_geometry_from_text(phrase)
    print(output_dict)
    return render_template('process.html', name=first_name,phrase=phrase, result=output_dict)
    
    
    
    
	
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

def content_loss(base, combination):
    return K.sum(K.square(combination - base))


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

def get_text_sentiment(input_text):

    sentiment_analyzer = SentimentIntensityAnalyzer()

    sentiment_score = sentiment_analyzer.polarity_scores(input_text)

    return sentiment_score	

def parse_file_to_dict(file_path):
    mapping_df = pd.read_excel(file_path, sheet_name='map')

    mapping_df['Keywords'] = mapping_df['Keywords'].apply(lambda x: x.split(','))
    mapping_df['Keywords'] = mapping_df['Keywords'].apply(lambda x: [keyword.strip() for keyword in x])

    mapping_df['Category'] = mapping_df['Category'].str.strip()
    mapping_df['Style'] = mapping_df['Style'].str.strip()

    mapping_df.set_index('Category', inplace=True)

    category_dict = mapping_df.to_dict(orient='index')

    return category_dict


def get_text_category(input_text, category_dict):

    for category in category_dict.keys():

        category_list = category_dict[category]['Keywords']

        for keyword in category_list:

            if keyword in input_text:

                output_dict = {'Category': category}

                output_dict.update({'Style': category_dict[category]['Style']})

                return output_dict

    return None


def get_sentiment_geometry_from_text(input_text):

    file_path = './data/category_mapping.xlsx'

    category_dict = parse_file_to_dict(file_path)

    text_sentiment_dict = {'Sentiment': get_text_sentiment(input_text)}

    text_geometry_dict = get_text_category(input_text, category_dict)

    output_dict = text_sentiment_dict

    if text_geometry_dict is not None:
        output_dict.update(text_geometry_dict)

    return output_dict


# Run the application
app.run(debug=True)



class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()


for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # Save current generated image
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))




# Content image
plt.imshow(load_img(target_image_path, target_size=(img_height, img_width)))
plt.figure()

# Style image
plt.imshow(load_img(style_reference_image_path, target_size=(img_height, img_width)))
plt.figure()

# Generate image
plt.imshow(img)
plt.show()