import numpy as np
from numba import cuda, float64
import math
import operator

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

        common = pointsAsArray[index][pIndex]


        # if(common[0] <= 14 and common[1] <= 14):
        #     cuda.atomic.add(vectors[index], anglesIntervals*propsIntervals + 0, 1)
        # elif(common[0] > 14 and common[1] <= 14):
        #     cuda.atomic.add(vectors[index], anglesIntervals*propsIntervals + 1, 1)
        # elif(common[0] > 14 and common[1] > 14):
        #     cuda.atomic.add(vectors[index], anglesIntervals*propsIntervals + 2, 1)
        # else:
        #     cuda.atomic.add(vectors[index], anglesIntervals*propsIntervals + 3, 1)


        # Current, Other
        aux2 = cuda.local.array((2,2),float64)
        current = aux2[0]
        other = aux2[1]

        for j in range(length):
            if (j != pIndex):
                current[0] = pointsAsArray[index][j][0] - common[0]
                current[1] = pointsAsArray[index][j][1] - common[1]

                # Angle, normCurrent, normOther
                aux = cuda.local.array((3),float64)
                angle = aux[0]
                normCurrent = aux[1]
                normOther = aux[2]

                # Norma del punto acutal
                normCurrent = math.sqrt(math.pow(current[0],2)+math.pow(current[1],2))
                
                for i in range(length):
                    if(i != pIndex and i != j):
                        other[0] = pointsAsArray[index][i][0] - common[0]
                        other[1] = pointsAsArray[index][i][1] - common[1]

                        
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


def convertImgPointsToNpArray(imagesAsPoints):
        # 80 puntos maximo, 2 ejes por punto
    pointsAsArray = np.zeros((len(imagesAsPoints),80,2))
    pointsLength = np.zeros(len(imagesAsPoints))

    for i in range(len(imagesAsPoints)):
        l = len(imagesAsPoints[i])
        pointsLength[i]=l
        for j in range(l):
            pointsAsArray[i][j]=imagesAsPoints[i][j]
    
    {pointsAsArray, pointsLength}

@cuda.jit(debug=True)
def distance(vectors, dist, query, n):
    index = cuda.blockIdx.x

    for i in range(n-4):
        dist[index] += math.pow(vectors[index][i] - query[i], 2)
    for i in range(n-4,n):
        dist[index] += math.pow(((vectors[index][i] - query[i]) * 1), 2)

    dist[index] = math.sqrt(dist[index])



def calculateDistances(vectors, samples, query):
    keys = range(samples)
    d = dict.fromkeys(keys)

    dist = np.zeros((samples),dtype=np.float)

    threadsperblock = 1
    blockspergrid = samples
    distance[blockspergrid, threadsperblock](vectors, dist, query, len(query))


   
    for i in range(samples):
        d[i] = dist[i]

    sort = sorted(d.items(), key=lambda kv:kv[1])

    res = np.zeros((15+1),dtype = np.int)
    i=0
    for (k,v) in sort:
        if(i< 15 + 1):
            res[i] = k
            i += 1
            # print("D: %f" % (v))
        else:
            break
    return res
