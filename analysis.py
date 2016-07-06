from data import *
import colorsys
from scipy import stats as sp
import numpy as np
import scipy.cluster.vq as vq
import random
import pcadata as pcadata

# Analysis methods
# Carl-Philip Majgaard
# CS 251
# Spring 2016

#get the range of a column
def dataRange(columns, data):
    colNum = []
    for col in columns: #grab the indexes of the columns
        colNum.append(data.getHeader(col))
    mins = data.matrix.min(axis = 0) #do some calculation
    maxs = data.matrix.max(axis = 0)

    forReturn = [] #set up our return
    for col in colNum:
        pair = [] #further setup
        pair.append(mins.item(col)) #add only requested columns
        pair.append(maxs.item(col))
        forReturn.append(pair) #add to return

    return forReturn

def mean(columns, data, matrix = None):
    if matrix != None:
        means = matrix.mean(axis = 0)

        forReturn = []
        for col in range(matrix.shape[1]):
            forReturn.append(means.item(col)) #filter
        return np.matrix(forReturn) #return
    else:
        colNum = []
        for col in columns:
            colNum.append(data.getHeader(col)) #grab the indexes
        means = data.matrix.mean(axis = 0) #calculate

        forReturn = []
        for col in colNum:
            forReturn.append(means.item(col)) #filter
        return np.matrix(forReturn) #return

def stdev(columns, data):
    colNum = []
    for col in columns:
        colNum.append(data.getHeader(col)) #grab indexes
    devs = data.matrix.std(axis = 0) #calculate

    forReturn = []
    for col in colNum:
        forReturn.append(devs.item(col)) #filter
    return forReturn #return

def mode(columns, data):
    colNum = []
    for col in columns:
        colNum.append(data.getHeader(col)) #grab indexes
    modes = stats.mode(data.matrix) #calculate

    forReturn = []
    for col in colNum:
        forReturn.append(modes[0].item(col)) #filter
    return forReturn #return

def translateColumns(data):
    a = data

    def operation(a): #define translation and scaling operation
        min = a.min()
        return (a-min) #return the processed 1D array

    return np.apply_along_axis(operation, 0, a) #apply the function to each col


def normalizeColumnsSeparately(columns, data):
    a = data.getData(columns) #grab columns

    def operation(a): #define translation and scaling operation
        min = a.min()
        max = a.max()
        return (a-min)/(max-min) #return the processed 1D array

    return np.apply_along_axis(operation, 0, a) #apply the function to each col

def normalizeColumnsTogether(columns, data):
    a = data.getData(columns) #grab columns
    min = a.min()
    max = a.max()

    return (a-min)/(max-min) #return processed matrix

#Use to generate color val
def pseudocolor(val):
    h = (float(val) / 1) * 120
    # convert hsv color (h,1,1) to its rgb equivalent
    r, g, b = colorsys.hsv_to_rgb(h/360, 1., 1.)
    r = r * 255
    g = g * 255
    b = b * 255
    return (g, r, b)

def linear_regression(data, ind, dep):
    pre_a = data.getData(ind)
    y = data.getData([dep])
    a = np.ones((pre_a.shape[0], pre_a.shape[1]+1))
    a[:,:-1] = pre_a
    AAinv = np.linalg.inv(np.dot(a.T, a))

    x = np.linalg.lstsq(a,y)
    b = x[0]
    n = y.shape[0]
    c = b.shape[0]
    df_e = n - c
    df_r = c - 1
    error = y - np.dot(a, b)
    sse = np.dot(error.T, error) / df_e
    stderr = np.sqrt( np.diagonal(sse[0,0] * AAinv))
    t = b.T / stderr
    p = 2*(1- sp.t.cdf(abs(t), df_e))
    r2 = 1 - error.var() / y.var()
    return b, sse, r2, t, p

def pca(data, headers, normalize = "True"):
    if normalize == "True":
        A = normalizeColumnsSeparately(headers, data)
    else:
        A = data.getData(headers)


    m = mean(headers, data, matrix = A)

    D = A - m

    U, S, V = np.linalg.svd(D, full_matrices=0)
    eigenval = (S*S)/(len(A)-1)

    vectors =  V

    projection  = ((V * D.T).T)

    return pcadata.PCAData(headers, projection, eigenval, vectors, m)


def kmeans_numpy( d, headers, K, whiten = True):
    A = d.getData(headers)

    # assign to W the result of calling vq.whiten on A
    W= vq.whiten(A)

    # assign to codebook, bookerror the result of calling vq.kmeans with W and K
    codebook, bookerror = vq.kmeans(W, K)

    # assign to codes, error the result of calling vq.vq with W and the codebook
    codes, error = vq.vq(W,codebook)

    # return codebook, codes, and error
    return codebook, codes, error


def kmeans_init(d, K, categories= ""):
    if categories != "":
        cats, labels = np.unique( np.asarray( categories.T ), return_inverse = True )
        means = np.matrix( np.zeros( (len(cats), d.shape[1]) ) )
        for i in range(len(cats)):
            means[i,:] = np.mean( d[labels==i, :], axis=0)
    else:
        means = np.matrix(np.zeros((K, d[0].size), dtype = np.float))
        maxes = d.max(0)
        mins = d.min(0)
        for i in range(d[0].size):
            for j in range(K):
                #print i
                means[j, i] = random.uniform(mins[0,i],maxes[0,i])
    return means

def kmeans_algorithm(A, means):
    # set up some useful constants
    MIN_CHANGE = 1e-7
    MAX_ITERATIONS = 100
    D = means.shape[1]
    K = means.shape[0]
    N = A.shape[0]

    # iterate no more than MAX_ITERATIONS
    for i in range(MAX_ITERATIONS):
        # calculate the codes
        codes, errors = kmeans_classify( A, means )

        # calculate the new means
        newmeans = np.zeros_like( means )
        counts = np.zeros( (K, 1) )
        for j in range(N):
            newmeans[codes[j,0],:] += A[j,:]
            counts[codes[j,0],0] += 1.0

        # finish calculating the means, taking into account possible zero counts
        for j in range(K):
            if counts[j,0] > 0.0:
                newmeans[j,:] /= counts[j, 0]
            else:
                newmeans[j,:] = A[random.randint(0,A.shape[0]),:]

        # test if the change is small enough
        diff = np.sum(np.square(means - newmeans))
        means = newmeans
        if diff < MIN_CHANGE:
            break

    # call classify with the final means
    codes, errors = kmeans_classify( A, means )

    # return the means, codes, and errors
    return (means, codes, errors)

def kmeans_classify(data, means):
    idxs = np.matrix(np.zeros((data.shape[0], 1), dtype = np.int))
    dist = np.matrix(np.zeros((data.shape[0], 1), dtype = np.float))
    for i in range(data.shape[0]):
        tempdists = []
        pt = data[i]
        for j in range(means.shape[0]):
            m = means[j]
            tempdists.append(np.linalg.norm(m-pt))
        inOrder = copy.copy(tempdists)
        inOrder.sort()
        dist[i,0] = inOrder[0]
        idxs[i,0] = tempdists.index(inOrder[0])
    return (idxs, dist)


def kmeans(d, headers, K, whiten=True, categories = ''):

    if whiten:
      # assign to W the result of calling vq.whiten on the data--> Normalized
        A= d.getData(headers)
        W= vq.whiten(A)
    # else
    else:
      # assign to W the matrix A--> unnormalized
        A = d.getData(headers)
        W = A


    # assign to codebook the result of calling kmeans_init with W, K, and categories
    codebook = kmeans_init(W, K, categories)


    # assign to codebook, codes, errors, the result of calling kmeans_algorithm with W and codebook
    codebook, codes, errors = kmeans_algorithm(W, codebook)

    # return the codebook, codes, and representation error
    return codebook, codes, errors

if __name__ == "__main__":
    filenames = ['pcatest.csv']
    for name in filenames:
        d = Data(name)
        pca(d, d.getHeaders())
