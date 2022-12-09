import json
import string
import numpy as np
import pandas as pd
import regex as re
import sympy
import scipy
import matplotlib
import sklearn
from scipy.spatial.distance import jaccard
from sklearn.metrics import jaccard_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


#loading the data
with open('TVs-all-merged.json') as f:
  data1 = json.load(f)

#getting the data to a list instead of dictionary
data2 = []
for key in data1.keys():
    dictionary = {}
    dictionary[key] = data1[key]
    data2.append(dictionary)

#bootstrap function
def bootstrapfunction(data2):
    n = len(data2)
    draws = set()
    rest = set([i for i in range(n)])
    for i in range(n):
        rndraw = np.random.randint(0,n)
        draws.add(rndraw)
        try:
           rest.remove(rndraw)
        except:
           pass

    draws = list(draws)
    data = []
    for k in range(len(draws)):
       data.append(data2[draws[k]])

    rest = list(rest)
    test = []
    for j in range(len(rest)):
        test.append(data2[rest[j]])

    return [data, test]

#obtaining model words
def modelwords(data):
    MW = set()
    mwtitle = set()
    mwvalues = set()
    for item in range(len(data)):
        for j in data[item].keys():
            elements = {}
            elements[item] = []
            title = data[item][j][0]["title"]
            mw = re.search("([a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*)", title)
            elements[item].append(mw.group().replace(" ",""))
            mwtitle.add(elements[item][0])
            MW.add(mw.group().replace(" ",""))
            features = data[item][j][0]["featuresMap"]
            for item2 in features.items():
                mw2 = re.search("([a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*)", item2[1])
                if mw2 != None:
                  mwvalues.add(mw2.group())
                  MW.add(mw2.group())
    return [MW, mwtitle, mwvalues]

#obtaining binary matrix of binary vectors
def binarymatrix(MW, data):
    bmatrix = np.zeros((len(MW), len(data)))
    k = 0

    for item in range(len(data)):
        for j in data[item].keys():
            k = k + 1
            print(k)
            features = data[item][j][0]["featuresMap"]
            print(features)
            for item2 in features.items():
                for i in range(len(MW)):
                    if list(MW)[i] == data[item][j][0]["title"] or list(MW)[i] == item2[1]:
                       bmatrix[i, k-1] = 1
                       print("YES", list(MW)[i])
    pd.DataFrame(bmatrix).to_csv("binarymatrix.csv")
    return bmatrix

#obtaining signature matrix
def signaturematrix(MW, data, bmatrix):
    n = 0.5*len(MW)
    S = np.matrix(np.ones((int(round(n)), len(data))) * 10000000000000000000000000000000000)

    for r in range(0, (len(bmatrix))):
        print("r is", r)
        for h in range(0, int(round(n))):
            a = np.random.randint(0, 2**32-1)
            b = np.random.randint(0, 2**32-1)
            f = sympy.nextprime(2**32-1)
            h_i = (a + b*r)%f
            for c in range(len(bmatrix.columns)):
               if bmatrix[r, c] == 1 and h_i < S[h, c]: #for when you have to compute it the first time (it is not saved as csv)
               #if bmatrix[str(c)][r] == 1 and h_i < S[h, c]: #for when bmatrix is in csv
                  S[h, c] = h_i
    print("Matrix S:")
    print(S)
    pd.DataFrame(S).to_csv("signaturematrix.csv")
    return S

#lsh
def listbuck(listbuckets, c, h_i):
    if h_i not in listbuckets.keys():
       listbuckets[h_i] = c
    elif type(listbuckets[h_i]) == list:
        listbuckets[h_i].append(c)
    else:
        lists = []
        lists.append(listbuckets[h_i])
        lists.append(c)

#obtaining the dissimilarity matrix
def distancematrix(MW, data, S):
    b = 120
    lb = 0
    #ub = int(round((len(MW)/b)))
    ub = int(round((len(S) / b)))
    listbuckets = {}

    for band in range(1, b):
        print("b is", band)
        a = np.random.randint(0, 10000000000)
        g = np.random.randint(0, 10000000000)
        f = sympy.nextprime(1000000000)
        for c in range(len(S.columns)):
            string = ""
            for s in range(lb, ub):
                if S[str(c)][s] > 0:
                   string = string + str(S[s,c]) #for when you have to compute it the first time (it is not saved as csv)
                   #string = string + str(S[str(c)][s]) #for when S is in csv
                   value = int(string)
                   h_i = (a + g * value) % f
                   listbuck(listbuckets, c, h_i) #at index h_i (bucket), get the product c
        lb = lb + int(round((len(MW) / b)))
        ub = ub + int(round((len(MW) / b)))

    D = np.matrix(np.ones(len(data), len(data))*10000)
    count = 0
    for h in range(len(listbuckets)):
        print("h is: ", h)
        index = listbuckets[h]
        if type(index) == list:
          for i in range(len(index)):
              for j in range(len(index)):
                  sig1 = S[:,i] #for when S is not in csv
                  sig2 = S[:,j] #for when S is not in csv
                  #sig1 = S[str(i)] #for when S is in csv
                  #sig2 = S[str(j)] #for when S is in csv
                  jaccard = jaccard_score(sig1, sig2)
                  count = count + 1
                  D[j,i] = 1-jaccard
                  D[i,j] = 1-jaccard

    pd.DataFrame(D).to_csv("dissimilaritymatrix.csv")
    return distancematrix, count

#clustering
def clustering(D):
    d = 0
    for i in range(len(D)):
        for j in range(len(D)):
            if D[i,j] > d and D[i,j] != 10000:
               d = D[i,j]

    clustering = AgglomerativeClustering(affinity="precomputed", linkage = "single", distance_threshold=d).fit(D)
    clusteringassignment = clustering.labels_
    n = len(clusteringassignment)
    candidatepairs = set()
    count = 0
    for i in range(n-1):
        for j in range(i+1, n):
            if clusteringassignment[i] == clusteringassignment[j]:
                candidatepairs.add((i,j))
                count = count + 1
    return candidatepairs, count

def realduplicates(data):
    count = 0
    truepairs = set()

    for item in range(len(data)):
        for j in data[item].keys():
            if len(data[item][j]["modelID"]) > 1:
               count = count + 1
               for i in range(len(data[item][j]["modelID"])):
                   truepairs.add((data[item][0][j]["modelID"]))
    return truepairs, count


#5 BOOTSTRAPS
meanf1 = 0
meanf1star = 0
for bootstrap in range(1,5):
    data = bootstrapfunction(data2)
    data = data[0]
    MW = modelwords(data)[0]
    bmatrix = binarymatrix(MW, data)
    print("Binary matrix is: ", bmatrix)
    S = signaturematrix(MW, data, bmatrix)
    print("Signature matrix is: ", S)
    D = distancematrix(MW, data, S)[0]
    print("Dissimilarity matrix is: ", D)

    pairs = clustering(D)
    realdup = realduplicates(data)

    #f1 score
    f1sc = f1_score(realdup[0], pairs[0])
    print("f1 score: ", f1sc)
    meanf1 = meanf1 + f1sc

    #pair completeness
    dfound = pairs[1]
    dreal = realdup[1]
    pcom = dfound / dreal
    print("pair completeness: ", pcom)

    #pair quality
    numcomparisons = distancematrix(MW, data, S)[1]
    pqual = dfound / numcomparisons
    print("pair quality: ", pqual)

    #f1* score
    f1scstar = f1_score(pcom, pqual)
    print("f1* score: ", f1scstar)
    meanf1star = meanf1star + f1scstar

print("FINAL F1 SCORE (AVERAGED ACROSS BOOTSTRAPS): ", meanf1/5)
print("FINAL F1* SCORE (AVERAGED ACROSS BOOTSTRAPS): ", meanf1star/5)
