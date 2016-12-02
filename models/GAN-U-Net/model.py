import settings
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, Deconvolution2D, Activation
from keras.optimizers import Adam
from keras.layers.core import Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras import backend as K

def load_model(path):
	# Implement a function that loads data that the transformer saved
	# return data
	pass


def save_model(model, path):
	# Implement a function that saves a model to path
	pass


def create_model(**kwargs):
	image_width = kwargs.get('image_width', settings.image_width)
	image_height = kwargs.get('image_height', settings.image_height)
	num_input_channels = kwargs.get('num_input_channels', settings.num_input_channels)

	generator = create_generator(**kwargs)
	discriminator = create_discriminator(**kwargs)

	gan_input = Input(shape=(num_input_channels, image_width, image_height))
	H = generator(gan_input)
	gan_V = discriminator(H)
	GAN = Model(gan_input, gan_V)

	loss = kwargs.get('loss', settings.loss)
	metrics = kwargs.get('metrics', settings.metrics)
	optimizer = kwargs.get('optimizer', settings.optimizer)
	learning_rate = kwargs.get('learning_rate', settings.learning_rate)

	# For custom loss and metrics functions
	if loss == 'dice_coef_loss':
		loss = dice_coef_loss

	if metrics == ['dice_coef']:
		metrics = [dice_coef]

	if optimizer == 'adam':
		optimizer = Adam(lr=learning_rate, beta_1=kwargs.get('beta_1', settings.beta_1))

	GAN.compile(loss=loss, optimizer=optimizer)
	GAN.summary()

	return GAN


def create_generator(**kwargs):
	image_width = kwargs.get('image_width', settings.image_width)
	image_height = kwargs.get('image_height', settings.image_height)
	num_input_channels = kwargs.get('num_input_channels', settings.num_input_channels)
	num_output_channels = kwargs.get('num_output_channels', settings.num_output_channels)
	batch_size = kwargs.get('batch_size', settings.batch_size)
	num_kernels = kwargs.get('num_kernels', settings.num_kernels)
	kernel_width = kwargs.get('kernel_width', settings.kernel_width)
	kernel_height = kwargs.get('kernel_height', settings.kernel_height)
	kernel_stride = kwargs.get('kernel_stride', settings.kernel_stride)
	relu_alpha = kwargs.get('relu_alpha', settings.relu_alpha)
	dropout_probability = kwargs.get('dropout_probability', settings.dropout_probability)

	K.set_image_dim_ordering('th') # use theano dimension ordering

	##### encoder
	encoder_input = Input(shape=(num_input_channels, image_width, image_height))

	### BLOCK 1
	encoder_1 = Convolution2D(nb_filter=num_kernels, 
	                            nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                            subsample=(kernel_stride, kernel_stride))(encoder_input)
	# Output is num_kernels x image_width/2 x image_height/2 


	### BLOCK 2
	filter_multiplier = 2 # used to increase the number of convolutional filters to use
	encoder_2 = LeakyReLU(alpha=relu_alpha)(encoder_1)
	encoder_2 = Convolution2D(nb_filter=num_kernels * filter_multiplier, 
	                            nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                            subsample=(kernel_stride, kernel_stride))(encoder_2)
	encoder_2 = BatchNormalization()(encoder_2)
	# Output is num_kernels*2 x image_width/4 x image_height/4

	### BLOCK 3
	filter_multiplier = 4
	encoder_3 = LeakyReLU(alpha=relu_alpha)(encoder_2)
	encoder_3 = Convolution2D(nb_filter=num_kernels * filter_multiplier, 
	                            nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                            subsample=(kernel_stride, kernel_stride))(encoder_3)
	encoder_3 = BatchNormalization()(encoder_3)
	# Output is num_kernels*4 x image_width/8 x image_height/8

	### BLOCK 4
	filter_multiplier = 8
	encoder_4 = LeakyReLU(alpha=relu_alpha)(encoder_3)
	encoder_4 = Convolution2D(nb_filter=num_kernels * filter_multiplier, 
	                            nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                            subsample=(kernel_stride, kernel_stride))(encoder_4)
	encoder_4 = BatchNormalization()(encoder_4)
	# Output is num_kernels*8 x image_width/16 x image_height/16

	### BLOCK 5
	filter_multiplier = 8
	encoder_5 = LeakyReLU(alpha=relu_alpha)(encoder_4)
	encoder_5 = Convolution2D(nb_filter=num_kernels * filter_multiplier, 
	                            nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                            subsample=(kernel_stride, kernel_stride))(encoder_5)
	encoder_5 = BatchNormalization()(encoder_5)
	# Output is num_kernels*8 x image_width/32 x image_height/32


	### BLOCK 6
	filter_multiplier = 8
	encoder_6 = LeakyReLU(alpha=relu_alpha)(encoder_5)
	encoder_6 = Convolution2D(nb_filter=num_kernels * filter_multiplier, 
	                            nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                            subsample=(kernel_stride, kernel_stride))(encoder_6)
	encoder_6 = BatchNormalization()(encoder_6)
	# Output is num_kernels*8 x image_width/64 x image_height/64


	### BLOCK 7
	filter_multiplier = 8
	encoder_7 = LeakyReLU(alpha=relu_alpha)(encoder_6)
	encoder_7 = Convolution2D(nb_filter=num_kernels * filter_multiplier, 
	                            nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                            subsample=(kernel_stride, kernel_stride))(encoder_7)
	encoder_7 = BatchNormalization()(encoder_7)
	# Output is num_kernels*8 x image_width/128 x image_height/128


	### BLOCK 8
	filter_multiplier = 8
	encoder_8 = LeakyReLU(alpha=relu_alpha)(encoder_7)
	encoder_8 = Convolution2D(nb_filter=num_kernels * filter_multiplier, 
	                            nb_row=1, nb_col=1, border_mode='same',
	                            # tensorflow errors when kernel size > input size
	                            subsample=(kernel_stride, kernel_stride))(encoder_8)
	encoder_8 = BatchNormalization()(encoder_8)
	# Output is num_kernels*8 x image_width/256 x image_height/256

	##### decoder

	### BLOCK 1
	output_width = int(image_width / 2**7)
	output_height = int(image_height / 2**7)
	_decoder_1 = Activation('relu')(encoder_8)
	output_shape = (batch_size, num_kernels * filter_multiplier, output_width, output_height)
	_decoder_1 = Deconvolution2D(nb_filter=num_kernels * filter_multiplier, 
	                                   nb_row=kernel_width, nb_col=kernel_height, border_mode='same',
	                                   output_shape=output_shape,
	                                   subsample=(kernel_stride, kernel_stride))(_decoder_1)
	_decoder_1 = BatchNormalization(axis=1)(_decoder_1)
	_decoder_1 = Dropout(dropout_probability)(_decoder_1)
	# skip connect encoder 7 and decoder 1
	decoder_1 = merge([_decoder_1, encoder_7], mode='concat', concat_axis=1)
	# Output is num_kernels*8*2 x image_width/128 x image_height/128


	### BLOCK 2
	output_width = int(image_width / 2**6)
	output_height = int(image_height / 2**6)
	output_shape = (batch_size,  num_kernels * filter_multiplier, output_width, output_height)
	_decoder_2 = Activation('relu')(decoder_1)
	_decoder_2 = Deconvolution2D(nb_filter=num_kernels * filter_multiplier, 
		                               nb_row=kernel_width, nb_col=kernel_height, border_mode='same',
		                               output_shape=output_shape,
		                               subsample=(kernel_stride, kernel_stride))(_decoder_2)
	_decoder_2 = BatchNormalization()(_decoder_2)
	_decoder_2 = Dropout(dropout_probability)(_decoder_2)
	# skip connect encoder 6 and decoder 2
	decoder_2 = merge([_decoder_2, encoder_6], mode='concat', concat_axis=1)
	# Output is num_kernels*8*2 x image_width/64 x image_height/64

	### BLOCK 3
	output_width = int(image_width / 2**5)
	output_height = int(image_height / 2**5)
	output_shape = (batch_size,  num_kernels * filter_multiplier, output_width, output_height)
	_decoder_3 = Activation('relu')(decoder_2)
	_decoder_3 = Deconvolution2D(nb_filter=num_kernels * filter_multiplier, 
		                               nb_row=kernel_width, nb_col=kernel_height, border_mode='same',
		                               output_shape=output_shape,
		                               subsample=(kernel_stride, kernel_stride))(_decoder_3)
	_decoder_3 = BatchNormalization()(_decoder_3)
	_decoder_3 = Dropout(dropout_probability)(_decoder_3)
	# skip connect encoder 5 and decoder 3
	decoder_3 = merge([_decoder_3, encoder_5], mode='concat', concat_axis=1)
	# Output is num_kernels*8*2 x image_width/32 x image_height/32

	### BLOCK 4
	output_width = int(image_width / 2**4)
	output_height = int(image_height / 2**4)
	output_shape = (batch_size,  num_kernels * filter_multiplier, output_width, output_height)
	_decoder_4 = Activation('relu')(decoder_3)
	_decoder_4 = Deconvolution2D(nb_filter=num_kernels * filter_multiplier, 
		                               nb_row=kernel_width, nb_col=kernel_height, border_mode='same',
		                               output_shape=output_shape,
		                               subsample=(kernel_stride, kernel_stride))(_decoder_4)
	_decoder_4 = BatchNormalization()(_decoder_4)
	# _decoder_4 = Dropout(dropout_probability)(_decoder_4)
	# skip connect encoder 4 and decoder 4
	decoder_4 = merge([_decoder_4, encoder_4], mode='concat', concat_axis=1)
	# Output is num_kernels*8*2 x image_width/16 x image_height/16

	### BLOCK 5
	output_width = int(image_width / 2**3)
	output_height = int(image_height / 2**3)
	filter_multiplier = 4
	output_shape = (batch_size,  num_kernels * filter_multiplier, output_width, output_height)
	_decoder_5 = Activation('relu')(decoder_4)
	_decoder_5 = Deconvolution2D(nb_filter=num_kernels * filter_multiplier, 
		                               nb_row=kernel_width, nb_col=kernel_height, border_mode='same',
		                               output_shape=output_shape,
		                               subsample=(kernel_stride, kernel_stride))(_decoder_5)
	_decoder_5 = BatchNormalization()(_decoder_5)
	# _decoder_5 = Dropout(dropout_probability)(_decoder_5)
	# skip connect encoder 3 and decoder 5
	decoder_5 = merge([_decoder_5, encoder_3], mode='concat', concat_axis=1)
	# Output is num_kernels*4*2 x image_width/8 x image_height/8

	### BLOCK 6
	output_width = int(image_width / 2**2)
	output_height = int(image_height / 2**2)
	filter_multiplier = 2
	output_shape = (batch_size,  num_kernels * filter_multiplier, output_width, output_height)
	_decoder_6 = Activation('relu')(decoder_5)
	_decoder_6 = Deconvolution2D(nb_filter=num_kernels * filter_multiplier, 
		                               nb_row=kernel_width, nb_col=kernel_height, border_mode='same',
		                               output_shape=output_shape,
		                               subsample=(kernel_stride, kernel_stride))(_decoder_6)
	_decoder_6 = BatchNormalization()(_decoder_6)
	# _decoder_6 = Dropout(dropout_probability)(_decoder_6)
	# skip connect encoder 3 and decoder 6
	decoder_6 = merge([_decoder_6, encoder_2], mode='concat', concat_axis=1)
	# Output is num_kernels*2*2 x image_width/4 x image_height/4
    
	### BLOCK 7
	output_width = int(image_width / 2**1)
	output_height = int(image_height / 2**1)
	filter_multiplier = 1
	output_shape = (batch_size,  num_kernels * filter_multiplier, output_width, output_height)
	_decoder_7 = Activation('relu')(decoder_6)
	_decoder_7 = Deconvolution2D(nb_filter=num_kernels * filter_multiplier, 
		                               nb_row=kernel_width, nb_col=kernel_height, border_mode='same',
		                               output_shape=output_shape,
		                               subsample=(kernel_stride, kernel_stride))(_decoder_7)
	_decoder_7 = BatchNormalization()(_decoder_7)
	# _decoder_7 = Dropout(dropout_probability)(_decoder_7)
	# skip connect encoder 3 and decoder 7
	decoder_7 = merge([_decoder_7, encoder_1], mode='concat', concat_axis=1)
	# Output is num_kernels*2 x image_width/2 x image_height/2

	### BLOCK 7
	output_width = int(image_width / 2**1)
	output_height = int(image_height / 2**1)
	filter_multiplier = 1
	output_shape = (batch_size,  num_kernels * filter_multiplier, output_width, output_height)
	_decoder_7 = Activation('relu')(decoder_6)
	_decoder_7 = Deconvolution2D(nb_filter=num_kernels * filter_multiplier, 
		                               nb_row=kernel_width, nb_col=kernel_height, border_mode='same',
		                               output_shape=output_shape,
		                               subsample=(kernel_stride, kernel_stride))(_decoder_7)
	_decoder_7 = BatchNormalization()(_decoder_7)
	# _decoder_7 = Dropout(dropout_probability)(_decoder_7)
	# skip connect encoder 3 and decoder 7
	decoder_7 = merge([_decoder_7, encoder_1], mode='concat', concat_axis=1)
	# Output is num_kernels*2 x image_width/2 x image_height/2

	### BLOCK 8
	output_width = int(image_width / 2**0)
	output_height = int(image_height / 2**0)
	output_shape = (batch_size,  num_output_channels, output_width, output_height)
	decoder_8 = Activation('relu')(decoder_7)
	decoder_8 = Deconvolution2D(nb_filter=num_output_channels, 
		                               nb_row=kernel_width, nb_col=kernel_height, border_mode='same',
		                               output_shape=output_shape,
		                               subsample=(kernel_stride, kernel_stride))(decoder_8)
	# Output is num_output_channels x image_width x image_height

	decoder_output = Activation('tanh')(decoder_8)

	loss = kwargs.get('loss', settings.loss)
	metrics = kwargs.get('metrics', settings.metrics)
	optimizer = kwargs.get('optimizer', settings.optimizer)
	learning_rate = kwargs.get('learning_rate', settings.learning_rate)

	# For custom loss and metrics functions
	if loss == 'dice_coef_loss':
		loss = dice_coef_loss

	if metrics == ['dice_coef']:
		metrics = [dice_coef]

	if optimizer == 'adam':
		optimizer = Adam(lr=learning_rate, beta_1=kwargs.get('beta_1', settings.beta_1))

	generator = Model(encoder_input, decoder_output)
	generator.compile(loss=loss, optimizer=optimizer)

	return generator


def create_discriminator(**kwargs):
	image_width = kwargs.get('image_width', settings.image_width)
	image_height = kwargs.get('image_height', settings.image_height)
	num_kernels = kwargs.get('num_kernels', settings.num_kernels)
	kernel_width = kwargs.get('kernel_width', settings.kernel_width)
	kernel_height = kwargs.get('kernel_height', settings.kernel_height)
	kernel_stride = kwargs.get('kernel_stride', settings.kernel_stride)
	relu_alpha = kwargs.get('relu_alpha', settings.relu_alpha)
	num_input_channels = kwargs.get('num_input_channels', settings.num_input_channels)
	K.set_image_dim_ordering('th') # use theano dimension ordering


	discriminator_input = Input(shape=(num_input_channels, image_width, image_height))

	### BLOCK 1
	discriminator_1 = Convolution2D(nb_filter=num_kernels, 
	                  nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                  subsample=(kernel_stride, kernel_stride))(discriminator_input)
	discriminator_1 = LeakyReLU(alpha=relu_alpha)(discriminator_1)
	# Output is num_kernels x image_width/2 x image_height/2 

	### BLOCK 2
	filter_multiplier = 2 # used to increase the number of convolutional filters to use
	discriminator_2 = Convolution2D(nb_filter=num_kernels * filter_multiplier, 
	                  nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                  subsample=(kernel_stride, kernel_stride))(discriminator_1)
	discriminator_2 = BatchNormalization()(discriminator_2)
	discriminator_2 = LeakyReLU(alpha=relu_alpha)(discriminator_2)
	# Output is num_kernels*2 x image_width/4 x image_height/4

	### BLOCK 3
	filter_multiplier = 4 # used to increase the number of convolutional filters to use
	discriminator_3 = Convolution2D(nb_filter=num_kernels * filter_multiplier, 
	                  nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                  subsample=(kernel_stride, kernel_stride))(discriminator_2)
	discriminator_3 = BatchNormalization()(discriminator_3)
	discriminator_3 = LeakyReLU(alpha=relu_alpha)(discriminator_3)
	# Output is num_kernels*4 x image_width/8 x image_height/8

	### BLOCK 4
	filter_multiplier = 8 # used to increase the number of convolutional filters to use
	discriminator_4 = Convolution2D(nb_filter=num_kernels * filter_multiplier, 
	                  nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                  subsample=(kernel_stride, kernel_stride))(discriminator_3)
	discriminator_4 = BatchNormalization()(discriminator_4)
	discriminator_4 = LeakyReLU(alpha=relu_alpha)(discriminator_4)
	# Output is num_kernels*8 x image_width/16 x image_height/16

	### BLOCK 6
	filter_multiplier = 2 # used to increase the number of convolutional filters to use
	discriminator_5 = Convolution2D(nb_filter=1, 
	                  nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                  subsample=(kernel_stride, kernel_stride))(discriminator_4)
	discriminator_output = Activation('sigmoid')(discriminator_5)
	# Output is 1 x image_width/32 x image_height/32

	loss = kwargs.get('loss', settings.loss)
	metrics = kwargs.get('metrics', settings.metrics)
	optimizer = kwargs.get('optimizer', settings.optimizer)
	learning_rate = kwargs.get('learning_rate', settings.learning_rate)

	# For custom loss and metrics functions
	if loss == 'dice_coef_loss':
		loss = dice_coef_loss

	if metrics == ['dice_coef']:
		metrics = [dice_coef]

	if optimizer == 'adam':
		optimizer = Adam(lr=learning_rate, beta_1=kwargs.get('beta_1', settings.beta_1))

	discriminator = Model(discriminator_input, discriminator_output)
	discriminator.compile(loss=loss, optimizer=optimizer)

	return discriminator


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
