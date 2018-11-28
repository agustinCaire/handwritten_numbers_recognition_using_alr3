import numpy as np
from numba import cuda, float64
import math

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

        # Cuadrante -> 0 (rotacion vertical)
        # if(current[0] <= 0):
        #     cuda.atomic.add(vectors[index], anglesIntervals*propsIntervals + 0, 1)
        # else:
        #     cuda.atomic.add(vectors[index], anglesIntervals*propsIntervals + 1, 1)

        # if(current[1] <= 0):
        #     cuda.atomic.add(vectors[index], anglesIntervals*propsIntervals + 2, 1)
        # else:
        #     cuda.atomic.add(vectors[index], anglesIntervals*propsIntervals + 3, 1)


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



@cuda.jit(debug=True)
def distance(vectors, dist, query, n):
    index = cuda.blockIdx.x

    for i in range(n-4):
        dist[index] += math.pow(vectors[index][i] - query[i], 2)
    for i in range(n-4,n-2):
        dist[index] += math.pow(((vectors[index][i] - query[i]) * 5), 2)
    for i in range(n-2,n):
        dist[index] += math.pow(((vectors[index][i] - query[i]) * 5), 2)

    dist[index] = math.sqrt(dist[index])