import os
import sys
import json
import random
hamDicts={}
wordWeights={}
MAX_LEARN_ITERATIONS=20
spamFileCount=0
hamFileCount=0
spamWordCount=[]
hamWordCount=[]
learnedmodel={}
learnedmodel["bias"]=0
learnedmodel["spam_yLabel"]=1
learnedmodel["ham_yLabel"]=-1
learnedmodel["learntWords"]=[]
def computewordWeights(wordType,alpha,wordList):
    global learnedmodel,MAX_LEARN_ITERATIONS,wordWeights
    if wordType == "ham":
        if (learnedmodel["ham_yLabel"] * alpha )<=0:
            for words in wordList:
                wordWeights[words]+=(learnedmodel["ham_yLabel"]*wordList[words])
            learnedmodel["bias"]+=learnedmodel["ham_yLabel"]
    else:
        if (learnedmodel["spam_yLabel"] * alpha) <= 0:
            for words in wordList:
                wordWeights[words] += (learnedmodel["spam_yLabel"] * wordList[words])
            learnedmodel["bias"] += learnedmodel["spam_yLabel"]
def readSpamFile(fileName):
    global spamFileCount
    spamDicts = {}
    with open(fileName, 'r',encoding= "latin1") as f:
        for line in f:
            for word in line.split(" "):
                word=word.rstrip('\n').rstrip('\r')
                if (word.isnumeric() is False): #Check if to be removed...
                    if (word in spamDicts):
                        spamDicts[word]=spamDicts[word]+1
                    else:
                        spamDicts[word] = 1
                    if (word not in wordWeights):
                        wordWeights[word] = 0
    learnedmodel["learntWords"].append(("spamWords",spamDicts));
    spamFileCount+=1

def readHamFile(fileName):
    global hamFileCount
    hamDicts={}
    with open(fileName, 'r',encoding= "latin1") as f:
        for line in f:
            for word in line.split(" "):
                word=word.rstrip('\n').rstrip('\r')
                if (word.isnumeric() is False):
                    if word in hamDicts:
                        hamDicts[word]=hamDicts[word]+1
                    else:
                        hamDicts[word] = 1
                    if (word not in wordWeights):
                        wordWeights[word] = 0
    learnedmodel["learntWords"].append(("hamWords", hamDicts));
    hamFileCount+=1

def learnPerceptron():
    global learnedmodel,MAX_LEARN_ITERATIONS,wordWeights
    for i in range(1,MAX_LEARN_ITERATIONS,1):
        for wordTuple in learnedmodel["learntWords"]:
            alpha=0
            for words in wordTuple[1]:
                alpha=alpha+wordWeights[words] * wordTuple[1][words]
            alpha=alpha+learnedmodel["bias"]
            computewordWeights(wordTuple[0],alpha,wordTuple[1])
        random.shuffle(learnedmodel["learntWords"])

def generateModel(modelFileName):
    global learnedmodel,wordWeights
    jsonDump={}
    jsonDump["BIAS"]=learnedmodel["bias"]
    jsonDump["WEIGHTS"]=wordWeights
    jsonString = json.dumps(jsonDump, indent=4, sort_keys=True, ensure_ascii=False)
    with open(modelFileName, "w", encoding="latin1") as modelFile:
        modelFile.write(jsonString)


def listFiles(directoryPath):
    for root, dirs, files in os.walk(directoryPath):
        path = root.split('/')
        for file in files:
            if file.endswith(".txt"):
                if os.path.basename(root) == "spam":
                    readSpamFile(os.path.join(root,file))
                if os.path.basename(root) == "ham":
                 readHamFile(os.path.join(root,file))

if  len(sys.argv) != 2:
    print("Error: The input data path is NULL or empty\n")
    sys.exit(-1)

if  not sys.argv[1]:
    print("Error: The input data path is NULL or empty\n")
    sys.exit(-1)

listFiles(sys.argv[1])
learnPerceptron()
generateModel("permodel.txt")
sys.exit(0);
