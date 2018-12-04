import numpy as np
import math
from collections import Counter, defaultdict
import random

import cudaCode

# ALR 3 configuracion
angles = 10
proportions = 10

# cantidad de NN para consultas
nnCount = 3

# Calcula vectores utilizando ALR3
def calculateImagesVector(pointsAsArray,pointsLength, samples):

    centroids = np.zeros((samples,2),dtype=np.float)

    threadsperblock = 1
    blockspergrid = samples
    cudaCode.calculateCentroid[blockspergrid,threadsperblock](pointsAsArray,pointsLength,centroids)



    threadsperblock = 80
    blockspergrid = samples
    cudaCode.applyCentroid[blockspergrid,threadsperblock](pointsAsArray,pointsLength,centroids)
    
    lenOfVectors = angles*proportions + 4
    vectors = np.zeros((samples,lenOfVectors),dtype=np.float64)

    threadsperblock = len(pointsAsArray[0])
    blockspergrid = samples
    cudaCode.calculateSingleVector[blockspergrid, threadsperblock](pointsAsArray,pointsLength,angles,proportions,vectors)

    norms = np.zeros((samples),dtype=np.float)

    threadsperblock = 1
    blockspergrid = samples
    cudaCode.calculateNorms[blockspergrid,threadsperblock](vectors, lenOfVectors, norms)

    threadsperblock = lenOfVectors
    blockspergrid = samples
    cudaCode.applyNorm[blockspergrid,threadsperblock](vectors,lenOfVectors, norms)

    
    return vectors



# Extrae los puntos de una imagen binaria
def extractPointsFromImages(images):
    imagePoints = []

    def extractPointsFromImage(image):
        currentPoints = []
        for row in range(len(image)):
            for col in range(len(image[0])):
                if(image[row][col] == 0):
                    currentPoints.append((col, row))
        return currentPoints
    
    for i in range(len(images)):
        imagePoints.append(extractPointsFromImage(images[i]))

    return imagePoints

# Convertir una lista de lista de puntos a un array numpy para su uso con CUDA
def convertImgPointsToNpArray(imagesAsPoints):
        # 80 puntos maximo, 2 ejes por punto
    pointsAsArray = np.zeros((len(imagesAsPoints),80,2))
    pointsLength = np.zeros(len(imagesAsPoints))

    for i in range(len(imagesAsPoints)):
        l = len(imagesAsPoints[i])
        pointsLength[i]=l
        for j in range(l):
            pointsAsArray[i][j]=imagesAsPoints[i][j]
    
    return (pointsAsArray, pointsLength)


# Calcular NN de un vector consulta
def calculateNN(vectors, samples, query):
    keys = range(samples)
    d = dict.fromkeys(keys)

    dist = np.zeros((samples),dtype=np.float)

    threadsperblock = 1
    blockspergrid = samples
    cudaCode.distance[blockspergrid, threadsperblock](vectors, dist, query, len(query))
   
    for i in range(samples):
        d[i] = dist[i]

    sort = sorted(d.items(), key=lambda kv:kv[1])

    res = []
    i=0
    for (k,v) in sort:
        if(i< nnCount):
            res.append((k,v))
            i += 1
            # print("D: %f" % (v))
        else:
            break
    return res

# En base a la cantidad de ocurrencias de cada numero en NN
def labelByNNCount(labels,nn):
    dc = defaultdict(int)
    for x in range(0,len(nn)):
            dc[labels[nn[x][0]]] += 1

    resultLabel,count = Counter(dc).most_common(1)[0]  

    return resultLabel  

# En base a la media de distancia de cada numero en NN
def labelByNNMeanDistance(labels,nn):

    d = defaultdict(float)
    dc = defaultdict(int)
    for x in range(0,len(nn)):
            dc[labels[nn[x][0]]] += 1
            d[labels[nn[x][0]]] += nn[x][1]
    for x in range(10):
            if(dc[x] > 0):
                d[x] /= dc[x]

    minX=-10
    minValue=100000

    for k,v in d.items():
        if(v < minValue):
            minX = k
            minValue = v

    return minX

# Calcular el numero que es segun los Nearest Neighbors
def labelByNN(labels,nn):
    return labelByNNCount(labels,nn)
    # return labelByNNMeanDistance(labels,nn)




# Recalcular vectores
def calculateAllVectors():
    labels = np.load("data/labels.npy")
    skeletons = np.load("data/skeletons.npy")

    points = extractPointsFromImages(skeletons)
    samples = len(skeletons)
    (pointArray, pointLengths) = convertImgPointsToNpArray(points)

    vectors = calculateImagesVector(pointArray,pointLengths,samples)

    np.save("data/vectors.npy",vectors)

# Recalcular vectores del dataset MINIST
def calculateMinistVectors():
    pointsArray = np.load("data/ministPointsAsArray.npy")
    pointsLength = np.load("data/ministPointsAsArrayLength.npy")

    vectors = calculateImagesVector(pointsArray,pointsLength,60000)

    np.save("data/ministVectors.npy",vectors)

# Probar eficacia frente a los casos de prueba MINIST
def testMinist():
    vectors = np.load("data/vectors.npy")
    labels = np.load("data/labels.npy")

    ministVectors = np.load("data/ministVectors.npy")
    ministLabels = np.load("data/ministLabels.npy")

    errors = 0
    
    samples = 2000
    for i in range(samples):
        x = random.randint(0,60000)
        nn = calculateNN(vectors,len(vectors),ministVectors[x])
        resultLabel = labelByNN(labels,nn)
        if(resultLabel != ministLabels[x]):
            errors += 1
    
    return 100 - (errors / samples * 100)


def ministLabelCount():
    ministLabels = np.load("data/ministLabels.npy")

    counts = np.zeros(10)

    for i in range(len(ministLabels)):
            counts[ministLabels[i]] += 1
  
    return counts