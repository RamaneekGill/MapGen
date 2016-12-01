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
	K.set_image_dim_ordering('th') # use theano dimension ordering

	image_width = kwargs.get('image_width', settings.image_width)
	image_height = kwargs.get('image_height', settings.image_height)
	kernel_width = kwargs.get('kernel_width', settings.kernel_width)
	kernel_height = kwargs.get('kernel_height', settings.kernel_height)
	kernel_stride = kwargs.get('kernel_stride', settings.kernel_stride)
	padding = kwargs.get('padding', settings.padding)
	pool_size = kwargs.get('pool_size', settings.pool_size)
	relu_alpha = kwargs.get('relu_alpha', settings.relu_alpha)
	dropout_probability = kwargs.get('dropout_probability', settings.dropout_probability)

	num_input_channels = kwargs.get('num_input_channels', settings.num_input_channels)
	num_generator_channels = kwargs.get('num_generator_channels', settings.num_generator_channels)

	K.set_image_dim_ordering('th')

	##### GENERATOR
	generator_input = Input(shape=(num_input_channels, image_width, image_height))

	### BLOCK 1
	generator_1 = Convolution2D(nb_filter=num_generator_channels, 
	                            nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                            subsample=(kernel_stride, kernel_stride))(generator_input)
	generator_1 = BatchNormalization()(generator_1)
	# Output is num_generator_channels x image_width/2 x image_height/2 


	### BLOCK 2
	filter_multiplier = 2 # used to increase the number of convolutional filters to use
	generator_2 = LeakyReLU(alpha=relu_alpha)(generator_1)
	generator_2 = Convolution2D(nb_filter=num_generator_channels * filter_multiplier, 
	                            nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                            subsample=(kernel_stride, kernel_stride))(generator_2)
	generator_2 = BatchNormalization()(generator_2)
	# Output is num_generator_channels*2 x image_width/4 x image_height/4

	### BLOCK 3
	filter_multiplier = 4
	generator_3 = LeakyReLU(alpha=relu_alpha)(generator_2)
	generator_3 = Convolution2D(nb_filter=num_generator_channels * filter_multiplier, 
	                            nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                            subsample=(kernel_stride, kernel_stride))(generator_3)
	generator_3 = BatchNormalization()(generator_3)
	# Output is num_generator_channels*4 x image_width/8 x image_height/8

	### BLOCK 4
	filter_multiplier = 8
	generator_4 = LeakyReLU(alpha=relu_alpha)(generator_3)
	generator_4 = Convolution2D(nb_filter=num_generator_channels * filter_multiplier, 
	                            nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                            subsample=(kernel_stride, kernel_stride))(generator_4)
	generator_4 = BatchNormalization()(generator_4)
	# Output is num_generator_channels*8 x image_width/16 x image_height/16

	### BLOCK 5
	filter_multiplier = 8
	generator_5 = LeakyReLU(alpha=relu_alpha)(generator_4)
	generator_5 = Convolution2D(nb_filter=num_generator_channels * filter_multiplier, 
	                            nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                            subsample=(kernel_stride, kernel_stride))(generator_5)
	generator_5 = BatchNormalization()(generator_5)
	# Output is num_generator_channels*8 x image_width/32 x image_height/32


	### BLOCK 6
	filter_multiplier = 8
	generator_6 = LeakyReLU(alpha=relu_alpha)(generator_5)
	generator_6 = Convolution2D(nb_filter=num_generator_channels * filter_multiplier, 
	                            nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                            subsample=(kernel_stride, kernel_stride))(generator_6)
	generator_6 = BatchNormalization()(generator_6)
	# Output is num_generator_channels*8 x image_width/64 x image_height/64


	### BLOCK 7
	filter_multiplier = 8
	generator_7 = LeakyReLU(alpha=relu_alpha)(generator_6)
	generator_7 = Convolution2D(nb_filter=num_generator_channels * filter_multiplier, 
	                            nb_row=kernel_width, nb_col=kernel_height, border_mode='same', 
	                            subsample=(kernel_stride, kernel_stride))(generator_7)
	generator_7 = BatchNormalization()(generator_7)
	# Output is num_generator_channels*8 x image_width/128 x image_height/128


	### BLOCK 8
	filter_multiplier = 8
	generator_8 = LeakyReLU(alpha=relu_alpha)(generator_7)
	generator_8 = Convolution2D(nb_filter=num_generator_channels * filter_multiplier, 
	                            nb_row=1, nb_col=1, border_mode='same',
	                            # tensorflow errors when kernel size > input size
	                            subsample=(kernel_stride, kernel_stride))(generator_8)
	generator_8 = BatchNormalization()(generator_8)
	# Output is num_generator_channels*8 x image_width/256 x image_height/256

	##### DISCRIMINATOR

	### BLOCK 1
	output_width = int(image_width / 2**7)
	output_height = int(image_height / 2**7)
	_discriminator_1 = Activation('relu')(generator_8)
	output_shape = (batch_size, num_generator_channels * filter_multiplier, output_width, output_height)
	_discriminator_1 = Deconvolution2D(nb_filter=num_generator_channels * filter_multiplier, 
	                                   nb_row=kernel_width, nb_col=kernel_height,
	                                   output_shape=output_shape,
	                                   subsample=(kernel_stride, kernel_stride))(_discriminator_1)
	_discriminator_1 = BatchNormalization(axis=1)(_discriminator_1)
	_discriminator_1 = Dropout(dropout_probability)(_discriminator_1)
	# skip connect generator 7 and discriminator 1
	discriminator_1 = merge([_discriminator_1, generator_7], mode='concat', concat_axis=1)
	# Output is num_generator_channels*8*2 x image_width/128 x image_height/128


	### BLOCK 2
	output_width = int(image_width / 2**6)
	output_height = int(image_height / 2**6)
	output_shape = (batch_size,  num_generator_channels * filter_multiplier, output_width, output_height)
	_discriminator_2 = Activation('relu')(discriminator_1)
	_discriminator_2 = Deconvolution2D(nb_filter=num_generator_channels * filter_multiplier, 
		                               nb_row=kernel_width, nb_col=kernel_height,
		                               output_shape=output_shape,
		                               subsample=(kernel_stride, kernel_stride))(_discriminator_2)
	_discriminator_2 = BatchNormalization()(_discriminator_2)
	_discriminator_2 = Dropout(dropout_probability)(_discriminator_2)
	# skip connect generator 6 and discriminator 2
	discriminator_2 = merge([_discriminator_2, generator_6], mode='concat', concat_axis=1)
	# Output is num_generator_channels*8*2 x image_width/64 x image_height/64

	### BLOCK 3
	output_width = int(image_width / 2**5)
	output_height = int(image_height / 2**5)
	output_shape = (batch_size,  num_generator_channels * filter_multiplier, output_width, output_height)
	_discriminator_3 = Activation('relu')(discriminator_2)
	_discriminator_3 = Deconvolution2D(nb_filter=num_generator_channels * filter_multiplier, 
		                               nb_row=kernel_width, nb_col=kernel_height,
		                               output_shape=output_shape,
		                               subsample=(kernel_stride, kernel_stride))(_discriminator_3)
	_discriminator_3 = BatchNormalization()(_discriminator_3)
	_discriminator_3 = Dropout(dropout_probability)(_discriminator_3)
	# skip connect generator 5 and discriminator 3
	discriminator_3 = merge([_discriminator_3, generator_5], mode='concat', concat_axis=1)
	# Output is num_generator_channels*8*2 x image_width/32 x image_height/32

	### BLOCK 4
	output_width = int(image_width / 2**4)
	output_height = int(image_height / 2**4)
	output_shape = (batch_size,  num_generator_channels * filter_multiplier, output_width, output_height)
	_discriminator_4 = Activation('relu')(discriminator_3)
	_discriminator_4 = Deconvolution2D(nb_filter=num_generator_channels * filter_multiplier, 
		                               nb_row=kernel_width, nb_col=kernel_height,
		                               output_shape=output_shape,
		                               subsample=(kernel_stride, kernel_stride))(_discriminator_4)
	_discriminator_4 = BatchNormalization()(_discriminator_4)
	# _discriminator_4 = Dropout(dropout_probability)(_discriminator_4)
	# skip connect generator 4 and discriminator 4
	discriminator_4 = merge([_discriminator_4, generator_4], mode='concat', concat_axis=1)
	# Output is num_generator_channels*8*2 x image_width/16 x image_height/16

	### BLOCK 5
	output_width = int(image_width / 2**3)
	output_height = int(image_height / 2**3)
	filter_multiplier = 4
	output_shape = (batch_size,  num_generator_channels * filter_multiplier, output_width, output_height)
	_discriminator_5 = Activation('relu')(discriminator_4)
	_discriminator_5 = Deconvolution2D(nb_filter=num_generator_channels * filter_multiplier, 
		                               nb_row=kernel_width, nb_col=kernel_height,
		                               output_shape=output_shape,
		                               subsample=(kernel_stride, kernel_stride))(_discriminator_5)
	_discriminator_5 = BatchNormalization()(_discriminator_5)
	# _discriminator_5 = Dropout(dropout_probability)(_discriminator_5)
	# skip connect generator 3 and discriminator 5
	discriminator_5 = merge([_discriminator_5, generator_3], mode='concat', concat_axis=1)
	# Output is num_generator_channels*4*2 x image_width/8 x image_height/8

	### BLOCK 6
	output_width = int(image_width / 2**2)
	output_height = int(image_height / 2**2)
	filter_multiplier = 2
	output_shape = (batch_size,  num_generator_channels * filter_multiplier, output_width, output_height)
	_discriminator_6 = Activation('relu')(discriminator_5)
	_discriminator_6 = Deconvolution2D(nb_filter=num_generator_channels * filter_multiplier, 
		                               nb_row=kernel_width, nb_col=kernel_height,
		                               output_shape=output_shape,
		                               subsample=(kernel_stride, kernel_stride))(_discriminator_6)
	_discriminator_6 = BatchNormalization()(_discriminator_6)
	# _discriminator_6 = Dropout(dropout_probability)(_discriminator_6)
	# skip connect generator 3 and discriminator 6
	discriminator_6 = merge([_discriminator_6, generator_2], mode='concat', concat_axis=1)
	# Output is num_generator_channels*2*2 x image_width/4 x image_height/4
    
	### BLOCK 7
	output_width = int(image_width / 2**1)
	output_height = int(image_height / 2**1)
	filter_multiplier = 1
	output_shape = (batch_size,  num_generator_channels * filter_multiplier, output_width, output_height)
	_discriminator_7 = Activation('relu')(discriminator_6)
	_discriminator_7 = Deconvolution2D(nb_filter=num_generator_channels * filter_multiplier, 
		                               nb_row=kernel_width, nb_col=kernel_height,
		                               output_shape=output_shape,
		                               subsample=(kernel_stride, kernel_stride))(_discriminator_7)
	_discriminator_7 = BatchNormalization()(_discriminator_7)
	# _discriminator_7 = Dropout(dropout_probability)(_discriminator_7)
	# skip connect generator 3 and discriminator 7
	discriminator_7 = merge([_discriminator_7, generator_1], mode='concat', concat_axis=1)
	# Output is num_generator_channels*2 x image_width/2 x image_height/2

	### BLOCK 7
	output_width = int(image_width / 2**1)
	output_height = int(image_height / 2**1)
	filter_multiplier = 1
	output_shape = (batch_size,  num_generator_channels * filter_multiplier, output_width, output_height)
	_discriminator_7 = Activation('relu')(discriminator_6)
	_discriminator_7 = Deconvolution2D(nb_filter=num_generator_channels * filter_multiplier, 
		                               nb_row=kernel_width, nb_col=kernel_height,
		                               output_shape=output_shape,
		                               subsample=(kernel_stride, kernel_stride))(_discriminator_7)
	_discriminator_7 = BatchNormalization()(_discriminator_7)
	# _discriminator_7 = Dropout(dropout_probability)(_discriminator_7)
	# skip connect generator 3 and discriminator 7
	discriminator_7 = merge([_discriminator_7, generator_1], mode='concat', concat_axis=1)
	# Output is num_generator_channels*2 x image_width/2 x image_height/2

	### BLOCK 8
	output_width = int(image_width / 2**0)
	output_height = int(image_height / 2**0)
	output_shape = (batch_size,  num_output_channels, output_width, output_height)
	discriminator_8 = Activation('relu')(discriminator_7)
	discriminator_8 = Deconvolution2D(nb_filter=num_output_channels, 
		                               nb_row=kernel_width, nb_col=kernel_height,
		                               output_shape=output_shape,
		                               subsample=(kernel_stride, kernel_stride))(discriminator_8)
	# Output is num_output_channels x image_width x image_height

	discriminator_output = Activation('tanh')(discriminator_8)

	return Model(input=generator_input, output=discriminator_output)


