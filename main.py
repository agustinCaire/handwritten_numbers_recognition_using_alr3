import logic
# import logic3points as logic
import time
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
from collections import Counter, defaultdict


def printMatrixArray(matrixArray, cols, rows):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(24, 24),
                         sharex=True, sharey=True)
    ax = axes.ravel()

    for i in range(len(matrixArray)):
        ax[i].imshow(matrixArray[i], cmap=plt.cm.gray)
        ax[i].axis('off')

    fig.tight_layout()
    plt.show()

def printQuery(query):
    fig, axes = plt.subplots(nrows=2, ncols=int(n/2), figsize=(24, 24),
                         sharex=True, sharey=True)
    ax = axes.ravel()

    for i in range(len(matrixArray)):
        ax[i].imshow(matrixArray[i], cmap=plt.cm.gray)
        ax[i].axis('off')

    fig.tight_layout()
    plt.show()

def indexToMatrixArray(indexes,pointsArray,pointsLength):
    matrix = np.zeros((len(indexes),28,28))

    mi = 0
    for idx in indexes:
        for i in range(int(pointsLength[idx])):
            pointX = int(pointsArray[idx][i][1])
            pointY = int(pointsArray[idx][i][0])

            matrix[mi][pointX][pointY] = 255
            # matrix[int(idx)][pointsArray[int(idx)][i][0]][pointsArray[int(idx)][i][1]] = 255
        mi += 1

    return matrix


def printPoints(points,length):

    arr = np.zeros((28,28))
    
    image = Image.new('L',(28,28))

    px = image.load()
    for i in range(int(length)):
        px[points[i][0],points[i][1]] = 255

    image = image.resize((200,200), Image.ANTIALIAS)
    image.show()


def printSkeletons(start):
    skeletons = np.load("data/skeletons.npy")
    count = len(skeletons)

    printMatrixArray(skeletons,30,10)

def main():
    
    # logic.calculateMinistVectors()

    accuracy = logic.testMinist() 
    print(accuracy)

    # printSkeletons(0)

if __name__ == '__main__':
    main()
