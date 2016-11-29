import settings

def load_model(path):
	# Implement a function that loads data that the transformer saved
	# return data
	pass

def save_model(model, path):
	# Implement a function that saves a model to path
	pass

def create_model(**kwargs):
	# Implement a function that creates a model and returns it
	# return model
	image_width = kwargs.get('image_width', settings.image_width)
	image_height = kwargs.get('image_height', settings.image_height)
	kernel_width = kwargs.get('kernel_width', settings.kernel_width)
	kernel_height = kwargs.get('kernel_height', settings.kernel_height)
	kernel_stride = kwargs.get('kernel_stride', settings.kernel_stride)
	padding = kwargs.get('padding', settings.padding)
	relu_alpha = kwargs.get('relu_alpha', settings.relu_alpha)

	num_input_channels = kwargs.get('num_input_channels', settings.num_input_channels)
	num_generator_channels = kwargs.get('num_generator_channels', settings.num_generator_channels)


	# Unconditioned GAN inputs will be num_channels x image_width x image_height
	generator = Convolution2D(input_shape=(num_input_channels, image_width, image_height),
		                      nb_filter=num_generator_channels, 
		                      nb_row=kernel_width, nb_col=kernel_height, 
		                      subsample=(kernel_stride, kernel_stride))
	generator = ZeroPadding2D(padding=(padding, padding))(generator)
	generator = LeakyReLU(alpha=relu_alpha)(generator)
	# Output is num_generator_channels x image_width x image_height

	filter_multiplier = 2 # used to increase the number of convolutional filters to use

	generator = Convolution2D(input_shape=(num_generator_channels, image_width, image_height),
		                      nb_filter=num_generator_channels * filter_multiplier, 
		                      nb_row=kernel_width, nb_col=kernel_height, 
		                      subsample=(kernel_stride, kernel_stride))(generator)
	generator = ZeroPadding2D(padding=(padding, padding))(generator)
	generator = BatchNormalization()(generator)
	generator = LeakyReLU(alpha=relu_alpha)(generator)


	# nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH])

#    # e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
#    # -- input is (ngf) x 128 x 128
#    # e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    # -- input is (ngf * 2) x 64 x 64
    # e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    # -- input is (ngf * 4) x 32 x 32
    # e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    # -- input is (ngf * 8) x 16 x 16
    # e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    # -- input is (ngf * 8) x 8 x 8
    # e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    # -- input is (ngf * 8) x 4 x 4
    # e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    # -- input is (ngf * 8) x 2 x 2
    # e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    # -- input is (ngf * 8) x 1 x 1
    
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