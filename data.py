import csv
import os
import numpy as np

# Data API
# Carl-Philip Majgaard
# CS 251
# Spring 2016

class Data:

    def __init__(self,filename = None):
        #Set a bunch of fields
        self.rawHeaders = []
        self.rawTypes = []
        self.header2raw = {}
        self.rawPoints = []
        self.headersNumeric = []
        self.matrix = np.matrix([])
        self.header2matrix = {}
        self.enumDict = {}
        self.read(filename)

    # Read a file into our fields
    def read(self, filename):
        #Setting my working directory. Commented out for submission
        os.chdir("/Users/CarlPhilipMajgaard/Desktop/CS251/DataWiz/datawiz/")
        with open(filename, 'rU') as f: #Open file
            reader = csv.reader(f)
            for row in reader: #for each row
                if "#" not in row[0]: #if not a comment
                    self.rawPoints.append(row) #add to raw points

            self.rawHeaders = list(self.rawPoints.pop(0)) #add headers by pop
            self.rawTypes = list(self.rawPoints.pop(0)) #add types by pop

            for idx, header in enumerate(self.rawHeaders):
                self.header2raw[header] = idx #fill dictionary

        for idx, header in enumerate(self.rawHeaders):
            if self.rawTypes[idx].lower() == 'enum': #If enum
                self.headersNumeric.append(header) #append the header
                self.processEnum(idx, header) #call for enum processing

            elif self.rawTypes[idx].lower() == 'numeric':
                self.headersNumeric.append(header) #append the header

        matrixList = [] #will be turned into matrix
        for row in self.rawPoints:
            rowList = []
            for header in self.headersNumeric: #add only numeric columns
                rowList.append(float(row[self.header2raw[header]])) #add it
            matrixList.append(rowList) #append row to list

        self.matrix = np.matrix(matrixList) #make matrix

        for idx, header in enumerate(self.headersNumeric): #make new dict
            self.header2matrix[header] = idx #fill it

    #Processes columns with enums
    def processEnum(self, column, header):
        self.enumDict[header] = {} #dict of dicts - for multiple enum cols
        for row in self.rawPoints:
            if not self.enumDict[header].has_key(row[column]): #if new key
                self.enumDict[header][row[column]] = len(self.enumDict[header]) #make the key
                row[column] = self.enumDict[header][row[column]] #edit the cell
            else:
                row[column] = self.enumDict[header][row[column]] #edit the cell


    ## Accessors for matrix data ##

    def getHeaders(self):
        return self.headersNumeric

    def getHeader(self, header):
        return self.header2matrix[header]

    def getNumColumns(self):
        return len(self.getHeaders())

    def getRow(self, row):
        return self.matrix[row,:]

    def getValue(self, column, row):
        print(self.header2matrix)
        return self.matrix[row,self.header2matrix[column]]

    def getData(self, columns, min=None, max=None): #gets range
        colList = []
        for col in columns:
            if self.header2matrix.has_key(col):
                colList.append(self.header2matrix[col])
        if min != None and max != None:
            step1 = self.matrix[range(min,max),:] #some matric work.
            step2 = step1[:, colList] #must be in 2 steps, else numpy throws a fit
        else:
            step2 = self.matrix[:, colList]
        return step2

    ## End accessors for matrix data ##

    ## Accessors for raw data ##

    def getRawHeaders(self):
        return self.rawHeaders

    def getRawTypes(self):
        return self.rawTypes

    def getRawNumColumns(self):
        return len(self.rawPoints[0])

    def getRawNumRows(self):
        return len(self.rawPoints)

    def getRawRow(self, row):
        return self.rawPoints[row]

    def getRawValue(self, column, row):
        return self.rawPoints[row][self.header2raw[column]]


    def textDump(self):
        print(self.getRawHeaders())
        print(self.getRawTypes())
        for row in self.rawPoints:
            print(row)

    ## End accessors for raw data ##

#testing with supplied testdata1.csv
if __name__ == "__main__":
    d = Data("testdata2.csv")
    print(d.rawPoints)
    print(d.getRawRow(0))
    print(d.getRawValue('stringstuff', 0))

    #show that we can get numeric headers
    print(d.getHeaders())

    #show that we can get the number of columns
    print(d.getNumColumns())

    #show that we can the the contens of some rows
    print(d.getRow(0))
    print(d.getRow(1))

    #show that we can get a value
    # print(d.getValue("thing2", 2))
    #
    # #show that we can get all data for certain columns
    # print(d.getData(["thing2", "thing3"]))
    #
    # print("SEP")
    # print(d.getData(["thing2", "thing3"], 1, 5))
