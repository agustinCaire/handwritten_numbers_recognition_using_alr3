import numpy as np

import base64
from PIL import Image
import logic
from collections import Counter, defaultdict
import os
import io
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.util import invert
from skimage.morphology import skeletonize
from skimage.transform import resize
from skimage import img_as_bool

def imgToBase64(img):
    output = io.BytesIO()
    img = Image.fromarray(arrayToPILImage(img), 'L')
    img.save(output, format='PNG')
    imgBase64 = base64.b64encode(output.getvalue()).decode('utf-8')
    return imgBase64

def scale(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    maxDif = max(rmax-rmin,cmax-cmin)
    newImg=np.zeros((maxDif,maxDif,4))
    for i in range(maxDif):
        for j in range(maxDif):
            if(i + rmin < len(img) and j + cmin < len(img[0])):
               newImg[i][j] = img[i + rmin][j + cmin]

    # newImg=np.zeros((rmax-rmin,cmax-cmin,4))
    # for i in range(rmin,rmax):
    #     newImg[i-rmin] = img[i][cmin:(cmax)]
    
    return newImg

def centerInBox(img):
    scaled = resize(img, (20, 20))

    rows = np.any(scaled, axis=1)
    cols = np.any(scaled, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    newImg=np.zeros((24,24,4))
    if(rmax-rmin == 19):
        offset = (20-cmax) // 2
        for i in range(20):
            for j in range(20):
                if(2+j + offset < 24): 
                    newImg[2+i][2+j+offset] = scaled[i][j]
    
    if(cmax-cmin == 19):
        offset = (20-rmax) // 2
        for i in range(20):
            for j in range(20):
                if(2+i + offset < 24): 
                    newImg[2+i+offset][2+j] = scaled[i][j]

    return newImg


def arrayToPILImage(arr):
    res = np.ones((len(arr),len(arr[0])),dtype=np.uint8)
    length = 0

    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if(not arr[i][j]):
                res[i][j] = 0
                length +=1
            else:
                res[i][j] = 255

    return res

def imageToSkel(base64image):
    image = base64.b64decode(base64image)
    image = io.BytesIO(image)
    image = mpimg.imread(image, format='PNG')

    image = scale(image)
    image = centerInBox(image)

    numberMatrix = list()
    for i in range(len(image)):
        row = image[i]
        newRow = []
        for col in range(len(row)):
            if(row[col][3] > 0):
                newRow.append(1)
            else:
                newRow.append(0)
        numberMatrix.append(newRow)

    #aca se obtiene el esqueleto de la matriz de 1s y 0ros
    result = invert(skeletonize(np.array(numberMatrix, dtype=np.integer)))  

    res = np.ones((len(result),len(result[0])),dtype=np.uint8)
    length = 0

    for i in range(len(result)):
        for j in range(len(result[0])):
            if(not result[i][j]):
                res[i][j] = 0
                length +=1
            else:
                res[i][j] = 255

    return res


def processQuery(base64image):
    logic.calculateAllVectors()

    data = np.load("data/vectors.npy")
    labels = np.load("data/labels.npy")

    querySkel = imageToSkel(base64image)
    points = logic.extractPointsFromImages([querySkel])
    (pointsArray, pointsLenght) = logic.convertImgPointsToNpArray(points)
    vectors = logic.calculateImagesVector(pointsArray,pointsLenght,1)
    vector = vectors[0]

    samples = len(data)

    nn = logic.calculateNN(data,samples,vector)
    resultLabel = logic.labelByNN(labels,nn)
    
    skeletons = np.load("data/skeletons.npy")
    result = list()
    for i in range(len(nn)): 
        result.append(imgToBase64(skeletons[nn[i][0]]))
    querySkelBase64 = imgToBase64(querySkel)


    return (result,resultLabel, querySkelBase64)

    
    



def save(base64image,label):

    skel = imageToSkel(base64image)
    labels = list(np.load("data/labels.npy"))
    skeletons = list(np.load("data/skeletons.npy"))

    skeletons.append(skel)
    labels.append(label)

    # labels = labels[:len(labels)-1]
    # skeletons = skeletons[:len(skeletons)-1]

    np.save("data/labels.npy", labels)
    np.save("data/skeletons.npy", skeletons)

    logic.calculateAllVectors()

    return imgToBase64(skel)
