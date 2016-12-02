random_seed = 1337

dropout_rate = 0.1

learning_rate = 1e-5

batch_size = 32

num_kernels = 64

num_input_channels = 3

num_output_channels = 3

beta_1 = 0.5 # beta for Adam optimizer

lamda = 100 # weight for L1 regularization

image_width = 256

image_height = 256

kernel_height = 4

kernel_width = 4

kernel_stride = 2

relu_alpha = 0.2 # for LeakyReLU

dropout_probability = 0.5

loss = 'dice_coef_loss'

optimizer = 'adam'

metrics = ['dice_coef']
