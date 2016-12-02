from PIL import Image

def train():
    (x_train, y_train), (X_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    x_train = x_train.reshape((x_train.shape[0], 1) + x_train.shape[1:])

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
            generated_images = generator.predict(x_image_batch, verbose=0)
            # if conditional GAN:
            #   generated_images = concat(generated_imgages, y_image_batch)

            # These two lines below are for debugging
            # predict_real_images = discriminator.predict(y_image_batch, verbose=0)
            # predict_fake_images = discriminator.predict(generated_images, verbose=0)

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



