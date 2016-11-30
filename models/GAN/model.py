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
	_discriminator_1 = Activation('relu')(generator_8)
	output_width = int(image_width / 2**7)
	output_height = int(image_height / 2**7)
	output_shape = (batch_size, output_width, output_height, num_generator_channels * filter_multiplier)
	_discriminator_1 = Deconvolution2D(nb_filter=num_generator_channels * filter_multiplier, 
		                               nb_row=kernel_width, nb_col=kernel_height,
		                               output_shape=output_shape,
		                               subsample=(kernel_stride, kernel_stride))(_discriminator_1)
	_discriminator_1 = BatchNormalization()(_discriminator_1)
	_discriminator_1 = Dropout(dropout_probability)(_discriminator_1)
	# Output is num_generator_channels*8 x image_width/128 x image_height/128
    
    # d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    # -- input is (ngf * 8) x 2 x 2
    # d1 = {d1_,e7} - nn.JoinTable(2)
    # d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    # -- input is (ngf * 8) x 4 x 4
    # d2 = {d2_,e6} - nn.JoinTable(2)
    # d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    # -- input is (ngf * 8) x 8 x 8
    # d3 = {d3_,e5} - nn.JoinTable(2)
    # d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    # -- input is (ngf * 8) x 16 x 16
    # d4 = {d4_,e4} - nn.JoinTable(2)
    # d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    # -- input is (ngf * 4) x 32 x 32
    # d5 = {d5_,e3} - nn.JoinTable(2)
    # d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    # -- input is (ngf * 2) x 64 x 64
    # d6 = {d6_,e2} - nn.JoinTable(2)
    # d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    # -- input is (ngf) x128 x 128
    # d7 = {d7_,e1} - nn.JoinTable(2)
    # d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
    # -- input is (nc) x 256 x 256
    
    # o1 = d8 - nn.Tanh()
    
    # netG = nn.gModule({e1},{o1})