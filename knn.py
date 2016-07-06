# Bruce Maxwell
# Spring 2015
# CS 251 Project 8
#
# KNN class test
#

import sys
import data
import classifiers
import numpy as np
import analysis as an


def main(argv):
    '''Reads in a training set and a test set and builds two KNN
    classifiers.  One uses all of the data, one uses 10
    exemplars. Then it classifies the test data and prints out the
    results.
    '''

    # usage
    if len(argv) < 3:
        print 'Usage: python %s <training data file> <test data file> <optional training category file> <optional test category file>' % (argv[0])
        exit(-1)

    # read the training and test sets
    dtrain = data.Data(argv[1])
    dtest = data.Data(argv[2])

    # get the categories and the training data A and the test data B
    if len(argv) > 4:
        traincatdata = data.Data(argv[3])
        testcatdata = data.Data(argv[4])
        traincats = traincatdata.getData( [traincatdata.getHeaders()[0]] )
        testcats = testcatdata.getData( [testcatdata.getHeaders()[0]] )
        A = dtrain.getData( dtrain.getHeaders() )
        B = dtest.getData( dtest.getHeaders() )
    else:
        # assume the categories are the last column
        traincats = dtrain.getData( [dtrain.getHeaders()[-1]] )
        testcats = dtest.getData( [dtest.getHeaders()[-1]] )
        A = dtrain.getData( dtrain.getHeaders()[:-1] )
        B = dtest.getData( dtest.getHeaders()[:-1] )

    #traincats = an.translateColumns(traincats)
    #testcats = an.translateColumns(testcats)

    # create two classifiers, one using 10 exemplars per class
    knnclass = classifiers.KNN()

    print "Created Classifier, Building Now."
    # build the classifiers
    knnclass.build( A, traincats )
    print "Built! Now classifying."
    acats, alabels = knnclass.classify( A )
    print "Classified."
    unique, mapping = np.unique(np.array(traincats.T), return_inverse=True)
    unique2, mapping2 = np.unique(np.array(alabels.T), return_inverse=True)
    mtx = knnclass.confusion_matrix(np.matrix(mapping).T, np.matrix(mapping2).T)
    print "Training Confusion Matrix:"
    print knnclass.confusion_matrix_str(mtx)

    # use the classifiers on the test data
    bcats, blabels = knnclass.classify( B )
    unique, mapping = np.unique(np.array(testcats.T), return_inverse=True)
    unique2, mapping2 = np.unique(np.array(blabels.T), return_inverse=True)
    mtx1 = knnclass.confusion_matrix(np.matrix(mapping).T, np.matrix(mapping2).T)

    print "Test Confusion Matrix:"
    print knnclass.confusion_matrix_str(mtx1)

    dtest.addColumn("KNN Classification", bcats)
    dtest.toFile(filename="KNNclass.csv")

    return

if __name__ == "__main__":
    main(sys.argv)
