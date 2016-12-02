import os
import numpy as np
from skimage import io
from PIL import Image

def load_data(path):
    path = os.path.join(os.path.dirname(path), 'raw')
    file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    sat_arr = []
    map_arr = []

    print('loading data')
    for filename in file_names:
        if not filename.endswith('.png'):
            print('not png')
            continue
        if 'map' in filename:
            fp = os.path.join(path, filename)
            map_arr.append(io.imread(fp))
        elif 'satellite' in filename:
            fp = os.path.join(path, filename)
            sat_arr.append(io.imread(fp))

    return [sat_arr, map_arr]

def train(model, training_data, validation_data, **kwargs):
    x_train = training_data[0]
    y_train = training_data[1]

    # model is the GAN itself
    generator = model.layers[1]
    discriminator = model.layers[2]
    make_trainable(discriminator, False)
    
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(x_train.shape[0]/batch_size))

        # for each batch
        for index in range(int(x_train.shape[0]/batch_size)):

            x_image_batch = x_train[index*batch_size:(index+1)*batch_size]
            y_image_batch = y_train[index*batch_size:(index+1)*batch_size]
            generated_images = generator.predict(x_image_batch, verbose=1)
            # if conditional GAN:
            #   generated_images = concat(generated_imgages, y_image_batch)

            # These two lines below are for debugging
            # predict_real_images = discriminator.predict(y_image_batch, verbose=1)
            # predict_fake_images = discriminator.predict(generated_images, verbose=1)

            # Train discriminator
            make_trainable(discriminator, True)
            x = np.concatenate((y_image_batch, generated_images))
            y = [1] * batch_size + [0] * batch_size
            d_loss = discriminator.train_on_batch(x, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            make_trainable(discriminator, False)

            # Train generator
            g_loss = model.train_on_batch(
                x_image_batch, [1] * batch_size)
            print("batch %d g_loss : %f" % (index, g_loss))

            # Save weights every 9 indexes
            if index % 10 == 9:
                generator.save_weights('generator_weights', True)
                discriminator.save_weights('discriminator_weights', True)

        # Save a generated image every epoch
        image = combine_images(generated_images)
        image = deprocess(image)
        Image.fromarray(image.astype(np.uint8)).save(
            str(epoch)+"_"+str(index)+".png")


def get_metrics(model, data):
    # Implement a function that computes metrics given a trained model
    # and dataset and returns a dictionary containing the metrics
    # the dictionary should not be nested
    # metrics = {}

    # return metrics
    pass


def process(image):
    return image


def deprocess(image):
    return image


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[0, :, :]
    return image


def make_trainable(net, val):
    ''' Make the layers in the model trainable or non-trainable '''
    net.trainable = val
    for l in net.layers:
        l.trainable = val

