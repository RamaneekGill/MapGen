import os
from skimage import io

def load_data(path):
	# Implement a function that loads data that the transformer saved
	# return data
    fn = os.listdir(path)
    sat_arr = []
    map_arr = []

    i = 0
    while i < len(fn):
        split = fn[i].split('_')
        if split[1] == 'map':
            filename = os.path.join(path, fn[i])
            map_arr.append(io.imread(filename))
            split[1] = 'satellite'
            filename = os.path.join(path, '_'.join(split))
            sat_arr.append(io.imread(filename))
        i += 1

    return [sat_arr, map_arr]

def train(model, training_data, validation_data, **kwargs):
	# Implement a function that takes a model and data and trains it
	pass

def get_metrics(model, data):
	# Implement a function that computes metrics given a trained model
	# and dataset and returns a dictionary containing the metrics
	# the dictionary should not be nested
	# metrics = {}

	# return metrics

	pass