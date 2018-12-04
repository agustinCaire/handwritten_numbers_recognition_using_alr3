import numpy as np
import matplotlib.pyplot as plt
import logic


def printMatrixArray(matrixArray, cols, rows):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(24, 24),
                         sharex=True, sharey=True)
    ax = axes.ravel()

    for i in range(len(matrixArray)):
        ax[i].imshow(matrixArray[i], cmap=plt.cm.gray)
        ax[i].axis('off')

    fig.tight_layout()
    plt.show()


def printMinistSkeletons(start):
    skeletons = np.load("data/ministSkeletons.npy")

    printMatrixArray(skeletons[start:start+300],30,10)


def printSkeletons(start):
    skeletons = np.load("data/skeletons.npy")

    printMatrixArray(skeletons[start:start+300],30,10)


def main():

    ### Cantidad de digitos del dataset MINIST
    # print(logic.ministLabelCount())
    
    ### Recalcular vectores de MINIST
    # logic.calculateMinistVectors()

    ### Para ver la eficacia del algoritmo usando como consulta los digitos de MINIST
    # accuracy = logic.testMinist() 
    # print(accuracy)

    ### Mostar skeletons
    printSkeletons(0)
    # printMinistSkeletons(0)


if __name__ == '__main__':
    main()
