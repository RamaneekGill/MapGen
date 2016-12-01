import numpy as np 
import os
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage

def read_data(path):
	# Implement a function that reads data
	# return data
    fn = os.listdir(path)
    fn.sort()
    sat_arr = []
    map_arr = []

    i = 0
    while i < len(fn):
        split = fn[i].split('_')
        if split[1] == 'map':
            filename = os.path.join(path, fn[i])
            map_arr.append(io.imread(filename))
        else:
            filename = os.path.join(path, fn[i])
            sat_arr.append(io.imread(filename))
        i += 1

    return [sat_arr, map_arr]

def gradient_magnitude(filename, sigma):
    img = np.array(Image.open(filename).convert('L'))

    gx = np.zeros(img.shape)
    ndimage.gaussian_filter(img, sigma, (0,1), gx)

    gy = np.zeros(img.shape)
    ndimage.gaussian_filter(img, sigma, (1,0), gy)

    return np.hypot(gx, gy)

def save_magnitudes(inputdir, outputdir):
    fn = os.listdir(inputdir)
    fn.remove('.DS_Store')
    for i in range(len(fn)):
        split = fn[i].split('_')
        filename = os.path.join(inputdir, fn[i])
        if split[1] == 'satellite':
            magnitude = gradient_magnitude(filename, 0.5)
            result = Image.fromarray(magnitude * 2).convert('L')
            split[1] = 'magnitude'
            result.save(os.path.join(outputdir, '_'.join(split)))
            # Uncomment for side by side images 
            # plt.subplot(121)
            # plt.imshow(Image.open(filename).convert("RGB"), cmap = 'gray')
            # plt.title('Original Image') 
            # plt.xticks([])
            # plt.yticks([])
            # plt.subplot(122)
            # plt.imshow(magnitude, cmap = 'gray')
            # plt.title('Gradient Magnitude Image')
            # plt.xticks([])
            # plt.yticks([])
            # plt.show()
        else:
            img = Image.fromarray(np.array(Image.open(filename).convert("RGB"))).convert("RGB")
            img.save(os.path.join(outputdir, fn[i]))


def save_canny_dector(inputdir, outputdir):
    fn = os.listdir(inputdir)
    fn.remove('.DS_Store')
    for i in range(len(fn)):
        split = fn[i].split('_')
        filename = os.path.join(inputdir, fn[i])
        if split[1] == 'satellite':
            img = np.array(Image.open(filename).convert("RGB"))
            edges = cv2.Canny(img, 50, 300) # 30, 300 is good threshhold
            split[1] = 'canny'
            edges.save(os.path.join(outputdir, '_'.join(split)))
            # Uncomment for side by side images 
            # plt.subplot(121)
            # plt.imshow(img, cmap = 'gray')
            # plt.title('Original Image') 
            # plt.xticks([])
            # plt.yticks([])
            # plt.subplot(122)
            # plt.imshow(edges, cmap = 'gray')
            # plt.title('Canny Edge Image')
            # plt.xticks([])
            # plt.yticks([])
            # plt.show()
        else:
            img = Image.fromarray(np.array(Image.open(filename).convert("RGB"))).convert("RGB")
            img.save(os.path.join(outputdir, fn[i]))

def save_k_cluster(inputdir, outputdir):
    fn = os.listdir(inputdir)
    fn.remove('.DS_Store')
    for i in range(len(fn)):
        split = fn[i].split('_')
        filename = os.path.join(inputdir, fn[i])
        if split[1] == 'satellite':
            img = np.array(Image.open(filename).convert("RGB"))
            Z = img.reshape((-1,3))
            # convert to np.float32
            Z = np.float32(Z)
            # define criteria, number of clusters(K) and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 4
            ret,label,center = cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            result = center[label.flatten()]
            result = result.reshape((img.shape))

            # Uncomment for side by side images 
            plt.subplot(121)
            plt.imshow(img, cmap = 'gray')
            plt.title('Original Image') 
            plt.xticks([])
            plt.yticks([])
            plt.subplot(122)
            plt.imshow(result, cmap = 'gray')
            plt.title('k = ' + str(K) + ' Cluster Image')
            plt.xticks([])
            plt.yticks([])
            plt.show()
        # else:
        #     img = Image.fromarray(np.array(Image.open(filename).convert("RGB"))).convert("RGB")
        #     img.save(os.path.join(outputdir, fn[i]))

def save_harrison_corner(inputdir, outputdir):
    pass

def save_sift(inputdir, outputdir):
    pass

#save_magnitudes('../../data/raw/', '../../data/transformed/magnitude')
#save_canny_dector('../../data/raw/', '../../data/transformed/canny detector')
save_k_cluster('../../data/raw/', '../../data/transformed/k-cluster')


