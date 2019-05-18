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
import cv2
import random
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentiment_analyzer = SentimentIntensityAnalyzer()
app = Flask(__name__)
app.config['SECRET_KEY'] = 'our very hard to guess secretfir'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    generate_image_from_text_dict({'Sentiment': {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}, 'Category': 'sad', 'Style': 'circle donut', 'Color': [72, 35, 80, 31]})
    print("genero")
    return render_template('thank-you.html')	
	
@app.route('/process',methods=['GET', 'POST'])
def process():
    global img_height
    global img_width
    # This is the path to the image you want to transform.
    first_name = request.form['firstname']
    phrase = request.form['textarea']
    output_dict = get_sentiment_geometry_from_text(phrase)
    generate_image_from_text_dict(output_dict)
    target_image_path ='static/name_image.png'  
    # This is the path to the style image.
    style_reference_image_path = 'images/kandisky_1.jpg'
    width, height = load_img(target_image_path).size
    img_height = 200
    img_width = int(width * img_height / height)	

    target_image = K.constant(preprocess_image(target_image_path))
    style_reference_image = K.constant(preprocess_image(style_reference_image_path))
    # This placeholder will contain our generated image
    combination_image = K.placeholder((1, img_height, img_width, 3))
    # We combine the 3 images into a single batch
    input_tensor = K.concatenate([target_image,
                                  style_reference_image,
                                  combination_image], axis=0)
    # We build the VGG19 network with our batch of 3 images as input.
    # The model will be loaded with pre-trained ImageNet weights.
    model = vgg19.VGG19(input_tensor=input_tensor,
                        weights='imagenet',
                        include_top=False)
    print('Model loaded.')    
    # Dict mapping layer names to activation tensors
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    # Name of layer used for content loss
    content_layer = 'block5_conv2'
    # Name of layers used for style loss
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    # Weights in the weighted average of the loss components
    total_variation_weight = 1e-4
    style_weight = 1.
    content_weight = 0.025
    
    # Define the loss by adding all components to a `loss` variable
    loss = K.variable(0.)
    layer_features = outputs_dict[content_layer]
    target_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(target_image_features,
                                          combination_features)
    for layer_name in style_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layers)) * sl
    loss += total_variation_weight * total_variation_loss(combination_image)
    
    
    
    result_prefix = 'style_transfer_result'
    iterations = 1
    
    # Run scipy-based optimization (L-BFGS) over the pixels of the generated image
    # so as to minimize the neural style loss.
    # This is our initial state: the target image.
    # Note that `scipy.optimize.fmin_l_bfgs_b` can only process flat vectors.
    x = preprocess_image(target_image_path)
    x = x.flatten()
    
    
    # Get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, combination_image)[0]
    
    # Function to fetch the values of the current loss and the current gradients
    fetch_loss_and_grads = K.function([combination_image], [loss, grads])
    
    
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
        fname = "static/" + result_prefix + '_at_iteration_%d.png' % i
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
    #plt.show()	
    img_height2=img_height +20;
    img_width2=img_width +30;
    return render_template('process.html', name=first_name,phrase=phrase, result=output_dict, imgKand_width=img_width,imgKand_ht=img_height,
	imgKand_width2=img_width2,imgKand_ht2=img_height2)
    
    
    
    
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
	
def get_rand(max_value, min_value=0):
    r = random.randint(min_value, max_value - 1)
    # print(r,max_value)
    return r

def get_text_sentiment(input_text):

    sentiment_analyzer = SentimentIntensityAnalyzer()

    sentiment_score = sentiment_analyzer.polarity_scores(input_text)

    return sentiment_score	

def parse_file_to_dict(file_path):
    mapping_df = pd.read_excel(file_path, sheet_name='map')

    mapping_df['Keywords'] = mapping_df['Keywords'].apply(lambda x: x.split(','))
    mapping_df['Keywords'] = mapping_df['Keywords'].apply(lambda x: [keyword.strip() for keyword in x])

    mapping_df['Color'] = mapping_df['Color'].apply(lambda x: x.split(','))
    mapping_df['Color'] = mapping_df['Color'].apply(lambda x: [int(color.strip()) for color in x])

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

                output_dict.update({'Style': category_dict[category]['Style'],
                                    'Color': category_dict[category]['Color']})

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

	
def generate_image_from_text_dict(input_dict):
    # images are normalized between 0 (black) - 255 (white)
    MAX_VALUE = 255
    MAX_THICKNESS = 30

    WIDTH = 896
    HEIGHT = 504

    WIDTH_BORDER = 800
    HEIGHT_BORDER = 450
    OFFSET = 100
    # HEIGHT and WIDTH are inverted
    size = (HEIGHT, WIDTH, 4)

    def convert_rgb_bgr(color):
        if len(color) == 4:
            return (color[2], color[1], color[0], color[3])
        elif len(color) == 3:
            return (color[2], color[1], color[0])
        else:
            return -1
    # determine background
    compund = input_dict["Sentiment"]["compound"]
    if compund >= 0:
        background_color = convert_rgb_bgr((27, 54, 138, 54))
    else:
        background_color = convert_rgb_bgr((52, 0, 21, 20))

    val_rect = 10
    val_circle = 10
    val_lines = 10
    # circle params
    n_rects = int(val_rect * abs(compund))
    n_circles = int(val_circle * abs(compund))
    n_lines_parallel = int(val_lines * abs(compund))
    n_lines_bezier = 0
    n_lines_vertex = 0
    n_triangles = int(val_lines * abs(compund))

    dict_shapes = {
        "rects": n_rects,
        "circles": n_circles,
        "lines_parallel": n_lines_parallel,
        "lines_bezier": n_lines_bezier,
        "lines_vertex": n_lines_vertex,
        "triangles": n_triangles
    }

    colors = input_dict["Color"]

    # create the image matrix
    my_img = np.full(size, background_color, dtype="uint8")

    # iteration through all the dict to create shapes
    for k, v in dict_shapes.items():
        for i in range(v):

            # take a random color from the one given and random thickness
            if len(colors) > 1:
                random_color_index = get_rand(len(colors))
                current_color = colors[random_color_index]
            else:
                current_color = colors
            # thickness = get_rand(min(sizes[0], sizes[1]) / 4)
            # thickness = get_rand(MAX_THICKNESS)
            thickness = 2

            if k == "rects":

                # thickness = random.randint(0, 1) * -1
                thickness = -1

                vert_1 = (get_rand(WIDTH_BORDER - OFFSET, OFFSET), get_rand(HEIGHT_BORDER - OFFSET))
                vert_2 = (get_rand(WIDTH_BORDER - OFFSET, OFFSET), get_rand(HEIGHT_BORDER - OFFSET))
                cv2.rectangle(my_img, vert_1, vert_2, current_color, thickness)
                # remove
                # colors.remove(current_color)
            elif k == "circles":

                center = (get_rand(WIDTH_BORDER - OFFSET, OFFSET), get_rand(HEIGHT_BORDER - OFFSET, OFFSET))
                # radius = (get_rand(min(size[1], size[0]) / 2))
                radius = (get_rand(min(WIDTH_BORDER, size[0]) / 8))
                thickness = radius
                cv2.circle(my_img, center, radius, current_color, thickness)
                cv2.circle(my_img, center, int(radius * 0.01), background_color, thickness)
                # remove
                # colors.remove(current_color)

            elif k == "lines_parallel":
                lineThickness = 2
                # lines are always black
                current_color = (0, 0, 0)
                x_pos_1 = get_rand(WIDTH_BORDER - OFFSET, OFFSET)
                # to constraint the length of the lines
                x_pos_2 = x_pos_1 + get_rand(100, 50)

                y_pos_1 = get_rand(HEIGHT_BORDER - OFFSET, OFFSET)
                y_pos_2 = y_pos_1 + get_rand(100, 50)

                parallel_offset = 20

                cv2.line(my_img, (x_pos_1, y_pos_1), (x_pos_2, y_pos_2), current_color, lineThickness)
                cv2.line(my_img, (x_pos_1, y_pos_1 + parallel_offset), (x_pos_2, y_pos_2 + parallel_offset),
                         current_color, lineThickness)
                cv2.line(my_img, (x_pos_1, y_pos_1 + parallel_offset * 2), (x_pos_2, y_pos_2 + parallel_offset * 2),
                         current_color, lineThickness)
                cv2.line(my_img, (x_pos_1, y_pos_1 + parallel_offset * 3), (x_pos_2, y_pos_2 + parallel_offset * 3),
                         current_color, lineThickness)

                print("first points {},{}".format((x_pos_1, y_pos_1), (x_pos_2, y_pos_2)))
                print("second points {},{}".format((x_pos_1, y_pos_1 + parallel_offset),
                                                   (x_pos_2, y_pos_2 + parallel_offset)))
                print("third points {},{}".format((x_pos_1, y_pos_1 + parallel_offset * 2),
                                                  (x_pos_2, y_pos_2 + parallel_offset * 2)))
                print("fourth points {},{}".format((x_pos_1, y_pos_1 + parallel_offset * 3),
                                                   (x_pos_2, y_pos_2 + parallel_offset * 3)))

                print("NEW")
                m = (y_pos_1 - y_pos_2) / (x_pos_1 - x_pos_2)
                new_m = - ((x_pos_1 - x_pos_2) / (y_pos_1 - y_pos_2))
                b = int(y_pos_1 - m * x_pos_1)
                print(b)
                new_y_1 = int((new_m * x_pos_1) + (b + m * x_pos_1))
                new_y_2 = int((new_m * x_pos_2) + (b + m * x_pos_2))

                cv2.line(my_img, (x_pos_1 + 4, new_y_1), (x_pos_2 + 4, new_y_2), current_color, lineThickness)
                cv2.line(my_img, (x_pos_1 + 8, new_y_1), (x_pos_2 + 8, new_y_2), current_color, lineThickness)
                cv2.line(my_img, (x_pos_1 + 16, new_y_1), (x_pos_2 + 16, new_y_2), current_color, lineThickness)

                print("first points {},{}".format((x_pos_1 + 4, new_y_1), (x_pos_2 + 4, new_y_2)))
                print("second points {},{}".format((x_pos_1 + 8, new_y_1), (x_pos_2 + 8, new_y_2)))
                print("third points {},{}".format((x_pos_1 + 16, new_y_1), (x_pos_2 + 16, new_y_2)))


            elif k == "lines_vertex":
                lineThickness = 2
                # lines are always black
                current_color = (0, 0, 0)
                x_pos_1 = get_rand(WIDTH_BORDER - OFFSET, OFFSET)
                # to constraint the length of the lines
                x_pos_2 = x_pos_1 + get_rand(100, 50)

                y_pos_1 = get_rand(HEIGHT_BORDER - OFFSET, OFFSET)
                y_pos_2 = y_pos_1 + get_rand(100, 50)

                parallel_offset = 20
                # 50 %
                if get_rand(2, 0):
                    cv2.line(my_img, (x_pos_1, y_pos_1), (x_pos_2, y_pos_2), current_color, lineThickness)
                    cv2.line(my_img, (x_pos_1, y_pos_1), (x_pos_2, y_pos_2 + parallel_offset * 3),
                             current_color, lineThickness)
                else:
                    cv2.line(my_img, (x_pos_1, y_pos_1), (x_pos_2 + parallel_offset * 3, y_pos_2), current_color,
                             lineThickness)
                    cv2.line(my_img, (x_pos_1, y_pos_1), (x_pos_2, y_pos_2),
                             current_color, lineThickness)

            elif k == "triangles":
                triangle = np.array([[100, 300], [400, 800], [100, 900]], np.int32)
                cv2.fillConvexPoly(my_img, triangle, current_color)
                # remove
                # colors.remove(current_color)

            elif k == "lines_bezier":
                pass

    cv2.imwrite('static/name_image.png', my_img)
	

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
    fname = "static/"+result_prefix + '_at_iteration_%d.png' % i
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