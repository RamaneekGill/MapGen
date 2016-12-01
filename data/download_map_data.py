"""
Downloads map and its respective satellite image from Google Maps
This script will randomly sample N images from a two sets of latitudes and longitudes

To use: python3 download_map_data.py --num_images={int} --top_left={Latitude, Longitude} --bottom_right{Latitude, Longitude}
"""


import io
import os
import sys
import requests
from PIL import Image
from time import sleep
from random import uniform as random_float

def preproccess(bytes_data):
	# Crop out the Google logo
	image = Image.open(io.BytesIO(bytes_data))
	return image.crop((0, 0, 256, 256))
	

def get_imagery(url):
	# Request the image and preprocess it
	request = requests.get(url)
	return preproccess(request.content)


if __name__ == '__main__':
	if len(sys.argv) == 1:
		print(__doc__)
		exit()

	# Number of map images to download
	num_images = int(sys.argv[1][13:])

	# Location to mine data e.g. 'Toronto' or 'Vancouver'
	if len(sys.argv) != 3:
		print('You did not specify a bounding box defaulting to Toronto: \n '
			  'top_left = (43.75722,-79.5714167) and bottom_right = (43.624655,-79.2370207)')

		latitudes = [43.75722, 43.624655]
		longitudes = [-79.5714167, -79.2370207]
	else:
		top_left_coordinates = sys.argv[2][11:].split(',')
		bottom_right_coordinates = sys.argv[3][15:].split(',')
		latitudes = [int(top_left_coordinates[0]), int(bottom_right_coordinates[0])]
		longitudes = [int(top_left_coordinates[1]), int(bottom_right_coordinates[1])]
		
	min_latitude = min(latitudes)
	max_latitude = max(latitudes)
	min_longitude = min(longitudes)
	max_longitude = max(longitudes)

	print('Using latitude range: ', min_latitude, max_latitude)
	print('Using longitude range: ', min_longitude, max_longitude)

	# Base API URL
	base_url = 'https://maps.googleapis.com/maps/api/staticmap?'
	map_param = '&maptype=map'
	satellite_param = '&maptype=satellite'
	default_params = '&zoom=16&size=256x286&sensor=false&style=feature:all|element:labels|visibility:off'

	for i in range(500, 500 + num_images+1):
		# Sleep every 0.5 seconds so we don't get throttled
		sleep(0.5)

		local_latitude = random_float(min_latitude, max_latitude)
		local_longitude = random_float(min_longitude, max_longitude)

		location_param = 'center=' + str(local_latitude) + ',' + str(local_longitude)

		# Map image
		fname = str(i) + '_map_' + location_param + '.png'
		url = base_url + location_param + default_params + map_param
		img = get_imagery(url)
		img.save(os.path.join('raw/', fname))

		# Satellite image
		fname = str(i) + '_satellite_' + location_param + '.png'
		url = base_url + location_param + default_params + satellite_param
		img = get_imagery(url)
		img.save(os.path.join('raw/', fname))
