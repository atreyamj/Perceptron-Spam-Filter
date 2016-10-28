import os
import sys
import json
import math
jsonModel={}
biasValue=0
weightsDict={}
spamCount=0
hamCount=0

def buildModel(modelName):
    global spamProb,hamProb
    global biasValue,weightsDict,jsonModel
    with open(modelName, 'r', encoding="latin1") as modelFile:
        jsonModel=json.load(modelFile)
    biasValue= jsonModel["BIAS"]
    weightsDict = jsonModel["WEIGHTS"]


def getSpamProbability(spamProbability,fileContent):
    tokens = fileContent.split()
    spamWordCounts=[]
    spamCounts=[]
    spamProbab=[]
    for token in tokens:
        if token in spamDicts:
            spamWordCounts.append(spamDicts[token])
        else:
            if token in hamDicts:
                spamWordCounts.append(0)
            else:
                spamWordCounts.append(-1)
    for wordCounts in spamWordCounts:
        if wordCounts !=-1:
            spamCounts.append(wordCounts)
    for wordCounts in spamCounts:
        spamProbab.append(math.log((wordCounts + 1) / (jsonModel["spamWordTotal"] + jsonModel["uniqueWords"])))
    totalSpamProbab = sum(spamProbab) + spamProbability
    return totalSpamProbab

def doAlphaComputation(fileContent):
    global biasValue, weightsDict, jsonModel
    alpha=0
    tokens = fileContent.split()
    for token in tokens:
        if token in weightsDict:
            alpha+=weightsDict[token]
    alpha+=biasValue
    return alpha

def doClassifyDocument(fileName):
    filestream = open(fileName, "r", encoding="latin1")
    content = filestream.read()
    alpha=doAlphaComputation(content)
    if alpha > 0:
        return 0
    else:
        return 1

def getClassification(directoryPath):
    global spamCount,hamCount
    with open("nboutput.txt", "w", encoding="latin1") as nbout:
        for root, dirs, files in os.walk(directoryPath):
            path = root.split('/')
            for file in files:
                if file.endswith(".txt"):
                    classifcationValue=doClassifyDocument(os.path.join(root,file))
                    if classifcationValue ==1:
                        hamCount+=1
                        nbout.write("ham "+os.path.join(root,file)+"\n")
                    elif classifcationValue ==0:
                        spamCount += 1
                        nbout.write("spam "+os.path.join(root,file)+"\n")

def getPerformanceStatistics(outPutFileName):
    if not outPutFileName:
        return
    if len(outPutFileName) == 0:
        return
    results = []
    mappings = [[0, 0], [0, 0]]
    correctLabel=""
    predictLabel=""
    with open(outPutFileName) as outputModel:
        for inputLine in outputModel:
            index = inputLine.find(" ")
            if index <= len(inputLine):
                predictLabel = inputLine[0:index]
                pathOfFile = inputLine[index + 1:]
                nameOfFile = pathOfFile[pathOfFile.rfind("/") + 1:]
                if "ham" in nameOfFile:
                    correctLabel = "ham"
                else:
                    correctLabel = "spam"
                results = results + [(predictLabel, correctLabel)]

    for result in results:
        if (result[0] == result[1] and result[1] == "spam"):
            mappings[1][1] = mappings[1][1] + 1
        elif (result[0] == result[1] and result[1] == "ham"):
            mappings[0][0] = mappings[0][0] + 1
        elif (result[0] == "ham" and result[1] == "spam"):
            mappings[1][0] = mappings[1][0] + 1
        else:
            mappings[0][1] = mappings[0][1] + 1
    '''HANDLE DIVIDE BY ZERO'''
    hamPrecision = mappings[0][0] / (mappings[0][0] + mappings[1][0])
    spamPrecision = mappings[1][1] / (mappings[1][1] + mappings[0][1])
    hamRecall = mappings[0][0] / (mappings[0][0] + mappings[0][1])
    spamRecall = mappings[1][1] / (mappings[1][1] + mappings[1][0])
    hamFscore = 2 * hamPrecision * hamRecall / (hamPrecision + hamRecall)
    spamFscore = 2 * spamPrecision * spamRecall / (spamPrecision + spamRecall)
    print(hamPrecision)
    print(spamPrecision)
    print(hamRecall)
    print(spamRecall)
    print(hamFscore)
    print(spamFscore)

if  len(sys.argv) != 2:
    print("Error: The input data path is NULL or empty\n")
    sys.exit(-1)

if  not sys.argv[1]:
    print("Error: The input data path is NULL or empty\n")
    sys.exit(-1)

buildModel("permodel.txt")
getClassification(sys.argv[1])
getPerformanceStatistics("nboutput.txt")
sys.exit(0)