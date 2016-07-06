import numpy as np
import copy
from data import Data
import csv

class PCAData(Data):

    def __init__(self, ogDataHeaders, projectedData, eigenValues, eigenVectors, ogDataMeans):
        Data.__init__(self)
        self.eigenValues = eigenValues
        self.eigenVectors = eigenVectors
        self.meanDataValues = ogDataMeans
        self.projectedHeaders = ogDataHeaders
        self.matrix = projectedData
        self.rawHeaders = ["P"+`i` for i in range(len(ogDataHeaders))]
        self.rawTypes = ["numeric" for i in range(len(ogDataHeaders))]
        for idx, i in enumerate(ogDataHeaders):
            self.header2raw[i] = idx
        self.rawPoints = projectedData.tolist()
        self.rawPointsCopy = projectedData.tolist()
        self.headersNumeric = ogDataHeaders
        for idx, i in enumerate(self.rawHeaders):
            self.header2matrix[i] = idx

        print(self.projectedHeaders)

    def get_eigenvalues(self):
        return np.matrix(self.eigenValues.copy())

    def get_eigenvectors(self):
        return np.matrix(self.eigenVectors.copy())

    def get_data_means(self):
        return self.meanDataValues.copy()

    def get_data_headers(self):
        return copy.copy(self.projectedHeaders)

    def toFile(self, filename="dataDump.csv"):
        with open(filename, "wb") as csvfile:
            cwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            cwriter.writerow(self.rawHeaders)
            cwriter.writerow(self.rawTypes)
            print self.rawTypes
            print "yikes"
            dat = self.getData(self.rawHeaders).tolist()
            for row in dat:
                cwriter.writerow(row)
