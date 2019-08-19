
import numpy as np
import operator
from os import listdir

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        #read each line
        lineStr = fr.readline()
        for j in range(32):
            #transfer the former 32 words into int
            returnVect[0, 32*i + j] = int(lineStr[j])
    return returnVect


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(
            classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def handwritingClassTest():
    # category lable list of yangbendata
    hwLabels = []

    # yangbendata file list
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    # init yangbendata matrix
    trainingMat = np.zeros((m, 1024))
    # read all yangbendata to data-matrix
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr) # the number that training file refers to
        # save yangbendata into matrix
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)


    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    # loop-test each testFile
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        classNumStr = int(fileStr.split('_')[0])

        # get data-matrix
        vectorUnderTest = img2vector('digits/trainingDigits/%s' % fileNameStr)
        # classify
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("test sample: %d, classifier predict: %d, fact label: %d " %
                (i+1, classifierResult, classNumStr))

        if (classifierResult != classNumStr):
            errorCount += 1.0

        print("\nerror rate: %f" % (errorCount/float(mTest)))

handwritingClassTest()
