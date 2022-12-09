
The code works as follows. First, the data is loaded and turned into a list rather than a dictionary. Then, all functions are defined such that the bootstrapping can easily be done by iterating 5 times (5 bootstraps) over all the functions.

bootstrapfunction: takes as input the full data and performs bootstrap by drawing with replacement as many times as there are data points. If the same item is sampled twice, it is only kept once. This yields approximately 63% of the data

From now on, I will refer to the bootstrapped data as simply data.
modelwords: takes as input the data. It uses a regular expression to transform the words in the data such that words that should be identical are considered so (e.g., "30 inch" and "30 inches"). It outputs the full set of model words for the feature specifications and the product titles.

binarymatrix: takes as input the set of model words and the data. It computes the binary vector representations (stored in a matrix) of the data based on the model words in the feature specifications and the titles. 

signaturematrix: takes as input the set of model words, the binary matrix, and the data. Computes the signature matrix.

listbuck: takes as input an empty dictionary, an index, and a hash value. Creates a dictionary of all buckets, where each bucket is either a single product or a list, if candidate duplicates exist.

distancematrix: takes as input a set of model words, the signature matrix, and the data. This is the Locality Sensitive Hashing part. Divides the signature matrix into bands of equal number of rows. Creates the dissimilarity matrix based on 1-jaccard similarity. Also counts the number of comparisons made in order to create the dissimilarity matrix.

clustering: takes as input the dissimilarity matrix. Computes the threshold for the clustering to stop as the maximum non-infinity value in the dissimilarity matrix. Performs agglomerative single linkage clustering based on the dissimilarity matrix and the computed threshold. Also counts the number of duplicates found (number of pairs in each cluster).

realduplicates: takes as input the data. Computes the real duplicates based on the model ID and counts the number of real duplicates.

Finally, the algorithm performs 5 bootstraps and computes the average F1 and F1* score across the 5 bootstraps.
