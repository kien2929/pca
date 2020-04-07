from PIL import Image
import numpy as np
from scipy import spatial as sp


def PCA(path):
    image = Image.open(path).convert("L")
    width, height = image.size
    # resize image to 128 x 128
    if width != 128 or height != 128:
        image = image.resize((128, 128))
    # to numpy array
    D = np.array(image)
    D = D.flatten()

    pca = np.load("pca.npy")
    X = np.dot(D, pca)  # chieu sang pca
    # load ko gian vector pca
    Y = np.load("Y.npy")
    out = []
    index = []
    # tim 5 vector gan nhat
    for i in range(0, 5):
        tree = sp.KDTree(Y)
        c1, c2 = tree.query(X)  # tim vector gan nhat
        index.append(c2)  # add index cua vector gan nhat
        Y = np.delete(Y, c2, axis=0)  # xoa vector do khoi Y

    data = np.load("listImage.npy")
    index = np.array(index)
    for i in range(len(index)):
        out.append(data[index[i]])
    out = np.array(out)
    return out
