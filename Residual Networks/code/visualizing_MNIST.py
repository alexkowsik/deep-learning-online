from matplotlib import pyplot as plt
import numpy as np
import h5py
import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.keras import backend as K


# Code based on http://everettsprojects.com/2018/01/17/mnist-visualization.html
# pretrained MNIST Model: https://github.com/kj7kunal/MNIST-Keras


tf.compat.v1.disable_eager_execution()

# Set the matplotlib figure size
plt.rc('figure', figsize = (12.0, 12.0))

# Set the learning phase to false, the model is pre-trained.
K.set_learning_phase(False)
model = load_model('MNIST_keras_CNN.h5')

# Figure out what keras named each of the layers in the model
layer_dict = dict([(layer.name, layer) for layer in model.layers])
print(layer_dict.keys())

# A placeholder for the input images
input_img = model.input

# Dimensions of the images
img_width = 28
img_height = 28

# A constant size step function for gradient ascent
def constant_step(total_steps, step, step_size = 1):
    return step_size

# Define an initial divisor and decay rate for a varied step function
# This function works better than constant step for the output layer
init_step_divisor = 100
decay = 10

def vary_step(total_steps, step):
    return (1.0 / (init_step_divisor + decay * step))


# Function from the Keras blog that normalizes and scales
# a filter before it is rendered as an image
def normalize_image(x):
    # Normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # Clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # Convert to grayscale image array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Create a numpy array that represents the image of a filter
# in the passed layer output and loss functions. Based on the
# core parts of Francois Chollet's blog post.
def visualize_filter(layer_output, loss, steps = 256, step_fn = constant_step, input_initialization = 'random'):
    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 1)

    # Initialize the input image. Random works well for the conv layers,
    # zeros works better for the output layer.
    input_img_data = np.random.random(input_shape) * 255.
    if input_initialization == "zeros":
        input_img_data = np.zeros(input_shape)
    input_img_data = np.array(input_img_data).reshape(1, 28, 28, 1)

    # Run gradient ascent for the specified number of steps
    for i in range(steps):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step_fn(steps, i)

    final_img = input_img_data[0]

    return final_img

# Define a function that stitches the 28 * 28 numpy arrays
# together into a collage of filters for each layer.
def stitch_filters(layer_filters, y_img_count, x_img_count):
    margin = 2
    width = y_img_count * img_width + (y_img_count - 1) * margin
    height = x_img_count * img_height + (x_img_count - 1) * margin
    stitched_filters = np.zeros((width, height))

    # Fill the picture with our saved filters
    for i in range(y_img_count):
        for j in range(x_img_count):
            img = layer_filters[i * x_img_count + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height] = img

    return stitched_filters

# Start by visualizing the first convolutional layer
layer_name = 'conv1'
layer_filters = []

# For each filter in this layer
for i in range(32):
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, i])
    img = visualize_filter(layer_output, loss)
    layer_filters.append(img.reshape(28,28))

layer_filters = [normalize_image(image) for image in layer_filters]      
layer_image = stitch_filters(layer_filters, 4, 8)

plt.imshow(layer_image, cmap = 'gray')
plt.show()

# The second convolutional layer
layer_name = 'conv2'
layer_filters = []

# For each filter in this layer
for i in range(32):
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, i])
    img = visualize_filter(layer_output, loss)
    layer_filters.append(img.reshape(28,28))

layer_filters = [normalize_image(image) for image in layer_filters]
layer_image = stitch_filters(layer_filters, 4, 8)

plt.imshow(layer_image, cmap = 'gray')
plt.show()

# The third convolutional layer
layer_name = 'conv3'
layer_filters = []

# For each filter in this layer
for i in range(64):
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, i])
    img = visualize_filter(layer_output, loss)
    layer_filters.append(img.reshape(28,28))

layer_filters = [normalize_image(image) for image in layer_filters]
layer_image = stitch_filters(layer_filters, 8, 8)

plt.imshow(layer_image, cmap = 'gray')
plt.show()

# The final output layer of the model
output_filters = []

for i in range(10):
    output = model.output
    loss = K.mean(output[:, i])
    img = visualize_filter(output, loss,
                          steps = 4096,
                          step_fn = vary_step,
                          input_initialization = 'zeros')
    output_filters.append(img.reshape(28,28))

output_image_raw = stitch_filters(output_filters, 2, 5)

plt.imshow(output_image_raw, cmap = 'gray')
plt.show()