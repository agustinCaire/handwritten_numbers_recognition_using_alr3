import numpy as np
from numba import cuda, float64
import math
import operator

@cuda.jit()
def calculateCentroid(pointsAsArray, pointsAsArrayLength,centroids):
    index = cuda.blockIdx.x

    centroids[index][0] = 0
    centroids[index][1] = 0

    for i in range(pointsAsArrayLength[index]):
        centroids[index][0] += pointsAsArray[index][i][0]
        centroids[index][1] += pointsAsArray[index][i][1]

    centroids[index][0] /= pointsAsArrayLength[index]
    centroids[index][1] /= pointsAsArrayLength[index]

@cuda.jit()
def applyCentroid(pointsAsArray, pointsAsArrayLength, centroids):
    index = cuda.blockIdx.x
    pIndex = cuda.threadIdx.x

    if(pIndex < pointsAsArrayLength[index]):
        pointsAsArray[index][pIndex][0] -= centroids[index][0]
        pointsAsArray[index][pIndex][1] -= centroids[index][1]

@cuda.jit()
def calculateSingleVector(pointsAsArray, pointsAsArrayLength, anglesIntervals, propsIntervals,vectors):
    
    tX = cuda.threadIdx.x
    tY = cuda.threadIdx.y
    bDimX = cuda.blockDim.x
    bDimY = cuda.blockDim.y

    # Sample al que pertenece el thread
    index = cuda.blockIdx.x

    # Punto del sample al que pertenece el thread
    pIndex = tX

    # Cantidad de puntos
    length = pointsAsArrayLength[index]

    if(pIndex < length):
        current = pointsAsArray[index][pIndex]

        # Cuadrante
        # if(current[1] <= 0):
        #     cuda.atomic.add(vectors[index], anglesIntervals*propsIntervals + 0, 1)
        # else:
        #     cuda.atomic.add(vectors[index], anglesIntervals*propsIntervals + 1, 1)



        if(current[0] <= 0 and current[1] <= 0):
            cuda.atomic.add(vectors[index], anglesIntervals*propsIntervals + 0, 1)
        elif(current[0] > 0 and current[1] <= 0):
            cuda.atomic.add(vectors[index], anglesIntervals*propsIntervals + 1, 1)
        elif(current[0] > 0 and current[1] > 0):
            cuda.atomic.add(vectors[index], anglesIntervals*propsIntervals + 2, 1)
        else:
            cuda.atomic.add(vectors[index], anglesIntervals*propsIntervals + 3, 1)

 
        # Angle, normCurrent, normOther
        aux = cuda.local.array((3),float64)
        angle = aux[0]
        normCurrent = aux[1]
        normOther = aux[2]

        # Norma del punto acutal
        normCurrent = math.sqrt(math.pow(current[0],2)+math.pow(current[1],2))
        
        for i in range(length):
            if(i != pIndex):
                other = pointsAsArray[index][i]
                
                # Norma
                normOther = math.sqrt(math.pow(other[0],2)+math.pow(other[1],2))
                
                if(normOther < normCurrent):
                    normOther = normOther * 100 / normCurrent
                else:
                    normOther = normCurrent * 100 / normOther
                
                normOther = (normOther // (100 // propsIntervals))
                
                # Angulo
                angle = 0
                for a, b in zip(current, other):
                    angle += a * b
                
                angle = np.math.atan2(current[0]*other[1] - current[1]*other[0], angle)*180.0 / math.pi

                if(angle < 0):
                    angle +=360

                angle = (angle // (360 / anglesIntervals))

                # Añadir al histograma
                if(angle == anglesIntervals):
                    angle -=1
                if(normOther == propsIntervals):
                    normOther -=1
                
                cuda.atomic.add(vectors[index], int(angle) * propsIntervals + int(normOther), 1)

    cuda.syncthreads()

@cuda.jit
def calculateNorms(vectors,n, norms):
    index = cuda.blockIdx.x

    norms[index] = 0
    
    for i in range(n):
        norms[index] += math.pow(vectors[index][i],2)

    norms[index] = math.sqrt(norms[index])

@cuda.jit
def applyNorm(vectors, n, norms):
    index = cuda.blockIdx.x
    pIndex = cuda.threadIdx.x
    if(pIndex) < n:
        vectors[index][pIndex] /= norms[index]

def calculateImagesVector(pointsAsArray,pointsLength, samples, angles, proportions):

    centroids = np.zeros((samples,2),dtype=np.float)

    threadsperblock = 1
    blockspergrid = samples
    calculateCentroid[blockspergrid,threadsperblock](pointsAsArray,pointsLength,centroids)



    threadsperblock = 80
    blockspergrid = samples
    applyCentroid[blockspergrid,threadsperblock](pointsAsArray,pointsLength,centroids)
    
    lenOfVectors = angles*proportions + 4
    vectors = np.zeros((samples,lenOfVectors),dtype=np.float64)

    threadsperblock = len(pointsAsArray[0])
    blockspergrid = samples
    calculateSingleVector[blockspergrid, threadsperblock](pointsAsArray,pointsLength,angles,proportions,vectors)

    norms = np.zeros((samples),dtype=np.float)

    threadsperblock = 1
    blockspergrid = samples
    calculateNorms[blockspergrid,threadsperblock](vectors, lenOfVectors, norms)

    threadsperblock = lenOfVectors
    blockspergrid = samples
    applyNorm[blockspergrid,threadsperblock](vectors,lenOfVectors, norms)

    
    return vectors

def extractPointsFromImages(images):
    
    print("extracting points...")
    imagePoints = []

    def extractPointsFromImage(image):
        currentPoints = []
        for row in range(24):
            for col in range(24):
                if(image[row][col] == 0):
                    currentPoints.append((col, row))
        return currentPoints
    
    for i in range(len(images)):
        imagePoints.append(extractPointsFromImage(images[i]))

    np.save('points.npy', imagePoints)
    return imagePoints

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




@cuda.jit(debug=True)
def distance(vectors, dist, query, n):
    index = cuda.blockIdx.x

    for i in range(n-4):
        dist[index] += math.pow(vectors[index][i] - query[i], 2)
    for i in range(n-4,n):
        dist[index] += math.pow(((vectors[index][i] - query[i]) * 10), 2)

    dist[index] = math.sqrt(dist[index])



def calculateDistances(vectors, samples, query,nn):
    keys = range(samples)
    d = dict.fromkeys(keys)

    dist = np.zeros((samples),dtype=np.float)

    threadsperblock = 1
    blockspergrid = samples
    distance[blockspergrid, threadsperblock](vectors, dist, query, len(query))

   
    for i in range(samples):
        d[i] = dist[i]

    sort = sorted(d.items(), key=lambda kv:kv[1])

    res = []
    i=0
    for (k,v) in sort:
        if(i< nn):
            res.append((k,v))
            i += 1
            # print("D: %f" % (v))
        else:
            break
    return res
