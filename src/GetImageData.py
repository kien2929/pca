import os
from PIL import Image
import numpy.linalg as la
import glob
import numpy as np

# def GetMatrixD():
path = "E:/Study/Project/Python/xla/PCA_Face/images"
D = []
# get list file in data
allData = os.listdir(path)

data = []
# make list to get all image
for list_file in allData:
    data.append(path + "/" + list_file + "/*.png")

list_image = []
# get all image and to array numpy
for i in range(0, len(allData)):
    for fileName in glob.glob(data[i]):
        list_image.append(fileName)
        im = Image.open(fileName).convert('L')
        width, height = im.size
        # resize image to 128 x 128
        if width != 128 or height != 128:
            im = im.resize((128, 128))
        # to numpy array
        arr = np.array(im)
        arr = arr.flatten()

        D.append(arr)
list_image = np.array(list_image)
np.save("listImage", list_image)
D = np.array(D)
np.save("imageVector", D)

N, n = D.shape  # get size Nxn

mean_D = D.mean(axis=0)  # mean vector
U = D - mean_D
# matrix hiep bien
X = np.dot(U.T, U) / (N - 1)
# np.save("X", X)
e, EV = la.eigh(X)

# find index of 20 eig value max
index_max_eig_value = []
for i in range(0, 20):
    index = np.argmax(e, axis=0)
    index_max_eig_value.append(index)
    e = np.delete(e, index)

# get 20 eig vecotr -> pca
pca = []
for i in range(0, len(index_max_eig_value)):
    pca.append(EV[:, index_max_eig_value[i]])

pca = np.array(pca)
pca = pca.T
# ko gian pca voi 16kx20
np.save("pca", pca)
# giai tuong quan, chieu sang ko gian pca
Y = np.dot(D, pca)
# x1, x2 = Y.shape
np.save("Y", Y)  # database