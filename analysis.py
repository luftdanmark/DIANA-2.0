from data import *
from scipy import stats

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

def mean(columns, data):
    colNum = []
    for col in columns:
        colNum.append(data.getHeader(col)) #grab the indexes
    means = data.matrix.mean(axis = 0) #calculate

    forReturn = []
    for col in colNum:
        forReturn.append(means.item(col)) #filter
    return forReturn #return

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

if __name__ == "__main__":
    d = Data("GovernmentWineUK.csv") #Initializing data

    #show that we can print ranges for all cols incl. enums
    print("Print data range for all columns")
    print(dataRange(["Region","Vintage", "Consumption"], d))

    #show that we can print mean for all cols incl. enums
    print("Print mean for all columns")
    print(mean(["Region", "Vintage", "Consumption"], d))

    #show that we can print std for all cols incl. enums
    print("Print stdev for all columns")
    print(stdev(["Region","Vintage", "Consumption"], d))

    #show that we can print modes for all cols incl. enums
    print("Print modes for all columns")
    print(mode(["Region","Vintage", "Consumption"], d))

    #show that we can print normalization for all cols incl. enums
    print("Print columns normalized together")
    print(normalizeColumnsTogether(["Region","Vintage", "Consumption"], d))

    #show that we can print normalization for all cols serarately incl. enums
    print("Print columns normalized separately")
    print(normalizeColumnsSeparately(["Region","Vintage", "Consumption"], d))
