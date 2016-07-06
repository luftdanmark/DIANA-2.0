# TK Skeleton
# Modified by Carl-Philip Majgaard
# CS 251
# Spring 2016

import Tkinter as tk
import tkFileDialog
import tkFont as tkf
import math
import subprocess
import random
from view import *
from data import *
import analysis
from scipy import stats
import copy

#Dialog Class to be inherited from
class Dialog(tk.Toplevel):

    def __init__(self, parent, title = None):

        tk.Toplevel.__init__(self, parent)
        self.transient(parent)

        if title:
            self.title(title)

        self.parent = parent

        self.result = None

        body = tk.Frame(self)
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)

        self.buttonbox()

        self.grab_set()

        if not self.initial_focus:
            self.initial_focus = self

        self.protocol("WM_DELETE_WINDOW", self.cancel)

        self.geometry("+%d+%d" % (parent.winfo_rootx()+50,
                                  parent.winfo_rooty()+50))

        self.initial_focus.focus_set()

        self.wait_window(self)



    #
    # construction hooks

    def body(self, master):
        # create dialog body.  return widget that should have
        # initial focus.  this method should be overridden

        pass

    def buttonbox(self):
        # add standard button box. override if you don't want the
        # standard buttons

        box = tk.Frame(self)

        w = tk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
        w.pack(side=tk.LEFT, padx=5, pady=5)
        w = tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    #
    # standard button semantics

    def ok(self, event=None):

        if not self.validate():
            self.initial_focus.focus_set() # put focus back
            return

        self.withdraw()
        self.update_idletasks()

        self.apply()
        print "Cancelling"
        self.cancel()
        print "Canceled"
    def cancel(self, event=None):
        print "Cancelling"
        # put focus back to the parent window
        self.parent.focus_set()
        self.destroy()

    #
    # command hooks

    def validate(self):

        return 1 # override

    def apply(self):

        pass # override

class LinRegDialog(Dialog):

    def __init__(self, parent, data, title = 'Choose Variables'):

        tk.Toplevel.__init__(self, parent)
        self.transient(parent)

        if title:
            self.title(title)

        self.parent = parent

        self.result = None

        body = tk.Frame(self)
        self.initial_focus = self.body(body, data)
        body.pack(padx=5, pady=5)

        self.buttonbox()

        self.grab_set()

        if not self.initial_focus:
            self.initial_focus = self

        self.protocol("WM_DELETE_WINDOW", self.cancel)

        self.geometry("+%d+%d" % (parent.winfo_rootx()+50,
                                  parent.winfo_rooty()+50))

        self.initial_focus.focus_set()
        self.wait_window(self)

    def body(self, master, data):

        w = tk.Label(master, text="Independent Variable")
        w.pack()

        self.l0 = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
        for item in data.headersNumeric:
            self.l0.insert(tk.END, item)
        self.l0.config(height=6)
        self.l0.pack()

        self.l0.selection_set(0)
        self.l0.activate(0)

        w = tk.Label(master, text="Dependent Variable")
        w.pack()

        self.l1 = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
        for item in data.headersNumeric:
            self.l1.insert(tk.END, item)
        self.l1.config(height=6)
        self.l1.pack()

        self.l1.selection_set(0)
        self.l1.activate(0)

    def validate(self):
        self.result = self.l0.get(tk.ACTIVE), self.l1.get(tk.ACTIVE)
        return 1

class NameDialog(Dialog):

    def __init__(self, parent, length, title = 'Name this Analysis'):

        tk.Toplevel.__init__(self, parent)
        self.transient(parent)

        if title:
            self.title(title)

        self.parent = parent

        self.result = None

        body = tk.Frame(self)
        self.initial_focus = self.body(body, length)
        body.pack(padx=5, pady=5)

        self.buttonbox()

        self.grab_set()

        if not self.initial_focus:
            self.initial_focus = self

        self.protocol("WM_DELETE_WINDOW", self.cancel)

        self.geometry("+%d+%d" % (parent.winfo_rootx()+50,
                                  parent.winfo_rooty()+50))

        self.initial_focus.focus_set()

        self.wait_window(self)

    def body(self, master, length):
        w = tk.Label(master, text="Name:")
        w.pack()

        self.name = tk.Entry(master)
        self.name.pack()
        string = "PCA %d" % (length)
        self.name.insert(0, string)

    def validate(self):
        self.result = self.name.get()
        if self.result != "":
            return 1
        else:
            return 0


class ColDialog(Dialog):

    def __init__(self, parent, headers, title = 'Choose Columns'):
        self.headers = headers
        Dialog.__init__(self, parent, title=title)

    def body(self, master):
        headers = self.headers
        w = tk.Label(master, text="Choose Columns:")
        w.pack()

        self.colBox = tk.Listbox(master, selectmode=tk.EXTENDED)
        self.colBox.pack()

        for i in headers:
            self.colBox.insert(tk.END, i)

        self.var = tk.StringVar()

        c = tk.Checkbutton(
            master, text="Normalize", variable=self.var,
            onvalue="True", offvalue="False"
            )
        c.pack()
        c.select()
    def validate(self):
        self.result = (self.colBox.curselection(), self.var.get())
        if len(self.result) > 0:
            return 1
        else:
            return 0

class clusterDialog(Dialog):

    def __init__(self, parent, headers, title = 'Choose Columns'):
        self.headers = headers
        Dialog.__init__(self, parent, title=title)

    def body(self, master):
        headers = self.headers
        w = tk.Label(master, text="Choose Columns:")
        w.pack()

        self.colBox = tk.Listbox(master, selectmode=tk.EXTENDED)
        self.colBox.pack()

        w = tk.Label(master, text="Number of Clusters:")
        w.pack()

        self.clusterBox = tk.Entry(master)
        self.clusterBox.pack()

        for i in headers:
            self.colBox.insert(tk.END, i)

    def validate(self):
        self.result = list(self.colBox.curselection())
        self.clusters = self.clusterBox.get()
        if len(self.result) > 0 and self.clusters.isdigit():
            return 1
        else:
            return 0


class EigenDialog(Dialog):

    def __init__(self, parent, array, title = 'Eigen Info'):
        self.array = array
        Dialog.__init__(self, parent, title=title)

    def body(self, master):
        array = self.array
        for idx, i in enumerate(array):
            for idx2, j in enumerate(i):
                if type(j) is float:
                    e = tk.Label(master, text = "%.4f" % j)
                else:
                    e = tk.Label(master, text = j)
                e.grid(row=idx, column = idx2, sticky = tk.NSEW)
    def cancel(self, event=None):
        print "Cancelling"
        # put focus back to the parent window
        for i in self.winfo_children():
            i.grid_forget()
        self.parent.focus_set()
        self.destroy()
# create a class to build and manage the display
class DisplayApp:

    def __init__(self, width, height):

        # create a tk object, which is the root window
        self.root = tk.Tk()

        # width and height of the window
        self.initDx = width
        self.initDy = height
        self.dx = 2
        self.size = None
        self.color = None

        #Discrete colors
        self.colors = ["#ACFFCC","#FD1F74","#20FF57","#5F60E0","#BBF14E",
        "#C64288","#73B302","#F98DD9","#BACC0F","#A4B9FE","#1CA03A","#FE6340",
        "#27F1CF","#AD5407","#93CBF6","#578616","#E8CAFD","#87F898","#FA726D",
        "#42E39C","#ED8798","#11885D","#F2B656","#4F73A4","#EAFDD4","#A05551",
        "#CAD5F6","#47899A","#FBF2E6","#96749C"]

        # set up the geometry for the window
        self.root.geometry( "%dx%d+50+30" % (self.initDx, self.initDy) )

        # set the title of the window
        self.root.title("D.I.A.N.A. 2.0") # give it a sick name

        # set the maximum size of the window for resizing
        self.root.maxsize( 1600, 900 )

        # setup the menus
        self.buildMenus()

        # build the controls
        self.buildControls()

        # build the Canvas
        self.buildCanvas()

        # bring the window to the front
        self.root.lift()

        # - do idle events here to get actual canvas size
        self.root.update_idletasks()

        # now we can ask the size of the canvas
        print self.canvas.winfo_geometry()

        # set up the key bindings
        self.setBindings()

        # set up the application state
        self.objects = [] # list of data objects that will be drawn in the canvas
        self.data = None # will hold the raw data someday.
        self.baseClick = None # used to keep track of mouse movement
        self.view = View()
        self.endpoints = np.matrix( [[0,1,0,0,0,0],
                                     [0,0,0,1,0,0],
                                     [0,0,0,0,0,1],
                                     [1,1,1,1,1,1]] )
        self.axes = []
        self.regLines = []
        self.regEndpoints = None
        self.clearData()
        self.spacePoints = None
        self.scalefactor = 1
        self.pcaList = []
        self.pcacount = -1
        self.clusterCount = -1
        self.handleOpen()


    #Builds Axes
    def buildAxes(self):
        self.axes = []
        vtm = self.view.build()
        tend = vtm * self.endpoints
        self.axes.append(self.canvas.create_line(tend[0,0], tend[1,0],
                                                 tend[0,1], tend[1,1], fill='red'))
        self.axes.append(self.canvas.create_line(tend[0,2], tend[1,2],
                                                 tend[0,3], tend[1,3], fill='green'))
        self.axes.append(self.canvas.create_line(tend[0,4], tend[1,4],
                                                 tend[0,5], tend[1,5], fill='blue'))

    #Updates Axes and points
    def updateAxes(self):
        vtm = self.view.build()
        tend = vtm * self.endpoints
        for idx, line in enumerate(self.axes):
            self.canvas.coords(line, tend[0,idx*2], tend[1,idx*2],
                                     tend[0,idx*2+1], tend[1,idx*2+1])
        self.updateFits()

    def updateFits(self):
        if self.regEndpoints != None:
            vtm = self.view.build()
            tend = vtm * self.regEndpoints
            for i, line in enumerate(self.regLines):
                self.canvas.coords(line, tend[0,i*2], tend[1,i*2],
                                         tend[0,i*2+1], tend[1,i*2+1])

    def colorFix(self):
        colorD = self.colorOp.get()
        if colorD != "Ignore":
            self.color = analysis.normalizeColumnsTogether([colorD], self.data)
        colorDict = {}
        print self.color
        print np.unique(np.asarray(self.color)).size
        if np.unique(np.asarray(self.color)).size <= 30:
            colCount = 0
            for i, point in enumerate(self.objects):
                if str(self.color[i,0]) not in colorDict:
                    colorDict[str(self.color[i,0])] = self.colors[colCount]
                    colCount += 1
            if self.spacePoints != None:
                if len(self.objects) > 0:
                        for i, point in enumerate(self.objects):
                            self.canvas.itemconfigure(point, fill=colorDict[str(self.color[i,0])])

    def updatePoints(self):
        if self.spacePoints != None:
            vtm = self.view.build()
            tpoints = self.spacePoints * vtm.T

            l = tpoints[:,-2].tolist()
            sortz = [item for sublist in l for item in sublist]
            indices = sorted(range(len(sortz)), key=lambda k: sortz[k])
            if len(self.objects) > 0:
                if self.size == None:
                    dx = self.dx
                    for i in indices:
                        self.canvas.coords(self.objects[i], tpoints[i,0]-dx, tpoints[i,1]-dx, tpoints[i,0]+dx, tpoints[i,1]+dx)
                else:
                    for i in indices:
                        dx = self.size[i,0]*10 +2
                        self.canvas.coords(self.objects[i], tpoints[i,0]-dx, tpoints[i,1]-dx, tpoints[i,0]+dx, tpoints[i,1]+dx)

    #Adds points from file chosen in init
    def buildPoints(self, headers):

        self.clearData()
        print(len(self.axes))
        points = analysis.normalizeColumnsSeparately(headers[0:3], self.data)
        if headers[3] != "Ignore":
            self.size = analysis.normalizeColumnsSeparately([headers[3]], self.data)
        if headers[4] != "Ignore":
            self.color = analysis.normalizeColumnsTogether([headers[4]], self.data)
        vtm = self.view.build()
        self.spacePoints = np.ones((points.shape[0], 4))

        if headers[2] == "Ignore":
            self.spacePoints[:,:-2] = points
            self.spacePoints[:,-2] = np.zeros((points.shape[0]))
        else:
            self.spacePoints[:,:-1] = points

        tend = self.spacePoints * vtm.T


        l = tend[:,-2].tolist()
        sortz = [item for sublist in l for item in sublist]
        indices = sorted(range(len(sortz)), key=lambda k: sortz[k])
        self.objects = [None] * len(sortz)

        if self.size != None and self.color != None:
            for i in indices:
                dx = self.size[i,0]*10 +2
                mycolor = '#%02x%02x%02x' % analysis.pseudocolor(self.color[i,0])
                pt = self.canvas.create_oval( tend[i,0]-dx, tend[i,1]-dx, tend[i,0]+dx, tend[i,1]+dx,
                                            fill=mycolor, outline='')
                self.objects[i] = pt

        elif self.size != None:
            for i in indices:
                dx = self.size[i,0]*10 +2
                pt = self.canvas.create_oval( tend[i,0]-dx, tend[i,1]-dx, tend[i,0]+dx, tend[i,1]+dx,
                                            fill='black', outline='')
                self.objects[i] = pt

        elif self.color != None:
            for i in indices:
                dx = self.dx
                mycolor = '#%02x%02x%02x' % analysis.pseudocolor(self.color[i,0])
                pt = self.canvas.create_oval( tend[i,0]-dx, tend[i,1]-dx, tend[i,0]+dx, tend[i,1]+dx,
                                            fill=mycolor, outline='')
                self.objects[i] = pt
        else:
            for i in indices:
                dx = self.dx
                pt = self.canvas.create_oval( tend[i,0]-dx, tend[i,1]-dx, tend[i,0]+dx, tend[i,1]+dx,
                                            fill="black", outline='')
                self.objects[i] = pt

        #self.updatePoints()
        #self.updateAxes()

    #Adds points from file chosen in init
    def buildPCA(self, headers, data):

        self.clearData()
        print(len(self.axes))
        points = analysis.normalizeColumnsSeparately(headers[0:3], data)
        if len(data.headersNumeric) > 3:
            self.size = analysis.normalizeColumnsSeparately([headers[3]], data)
        if len(data.headersNumeric) > 4:
            self.color = analysis.normalizeColumnsTogether([headers[4]], data)
        vtm = self.view.build()
        self.spacePoints = np.ones((points.shape[0], 4))

        if len(data.headersNumeric) < 3:
            self.spacePoints[:,:-2] = points
            self.spacePoints[:,-2] = np.zeros((points.shape[0]))
        else:
            self.spacePoints[:,:-1] = points

        tend = self.spacePoints * vtm.T

        l = tend[:,-2].tolist()
        sortz = [item for sublist in l for item in sublist]
        indices = sorted(range(len(sortz)), key=lambda k: sortz[k])
        self.objects = [None] * len(sortz)

        if self.size != None and self.color != None:
            for i in indices:
                dx = self.size[i,0]*10 +2
                mycolor = '#%02x%02x%02x' % analysis.pseudocolor(self.color[i,0])
                pt = self.canvas.create_oval( tend[i,0]-dx, tend[i,1]-dx, tend[i,0]+dx, tend[i,1]+dx,
                                            fill=mycolor, outline='')
                self.objects[i] = pt

        elif self.size != None:
            for i in indices:
                dx = self.size[i,0]*10 +2
                pt = self.canvas.create_oval( tend[i,0]-dx, tend[i,1]-dx, tend[i,0]+dx, tend[i,1]+dx,
                                            fill='black', outline='')
                self.objects[i] = pt

        elif self.color != None:
            for i in indices:
                dx = self.dx
                mycolor = '#%02x%02x%02x' % analysis.pseudocolor(self.color[i,0])
                pt = self.canvas.create_oval( tend[i,0]-dx, tend[i,1]-dx, tend[i,0]+dx, tend[i,1]+dx,
                                            fill=mycolor, outline='')
                self.objects[i] = pt
        else:
            for i in indices:
                dx = self.dx
                pt = self.canvas.create_oval( tend[i,0]-dx, tend[i,1]-dx, tend[i,0]+dx, tend[i,1]+dx,
                                            fill="black", outline='')
                self.objects[i] = pt

        #self.updatePoints()
        #self.updateAxes()

    def buildLinearRegression(self, variables):
        normalized = analysis.normalizeColumnsSeparately(variables, self.data)
        points = np.ones((normalized.shape[0], 4))
        points[:,-2] = np.zeros((normalized.shape[0]))
        points[:,:-2] = normalized
        self.spacePoints = points

        vtm = self.view.build()
        tend = self.spacePoints * vtm.T

        if self.size != None and self.color != None:
            for i in range(0,tend.shape[0]):
                dx = self.size[i,0]*10 +2
                mycolor = '#%02x%02x%02x' % analysis.pseudocolor(self.color[i,0])
                pt = self.canvas.create_oval( tend[i,0]-dx, tend[i,1]-dx, tend[i,0]+dx, tend[i,1]+dx,
                                            fill=mycolor, outline='')
                self.objects.append(pt)

        elif self.size != None:
            for i in range(0,tend.shape[0]):
                dx = self.size[i,0]*10 +2
                pt = self.canvas.create_oval( tend[i,0]-dx, tend[i,1]-dx, tend[i,0]+dx, tend[i,1]+dx,
                                            fill='black', outline='')
                self.objects.append(pt)

        elif self.color != None:
            for i in range(0,tend.shape[0]):
                dx = self.dx
                mycolor = '#%02x%02x%02x' % analysis.pseudocolor(self.color[i,0])
                pt = self.canvas.create_oval( tend[i,0]-dx, tend[i,1]-dx, tend[i,0]+dx, tend[i,1]+dx,
                                            fill=mycolor, outline='')
                self.objects.append(pt)
        else:
            for i in range(0,tend.shape[0]):
                dx = self.dx
                pt = self.canvas.create_oval( tend[i,0]-dx, tend[i,1]-dx, tend[i,0]+dx, tend[i,1]+dx,
                                            fill="black", outline='')
                self.objects.append(pt)
        ind = self.data.getData([variables[0]]).tolist()
        dep = self.data.getData([variables[1]]).tolist()

        independent = [val for sublist in ind for val in sublist]
        dependent = [val for sublist in dep for val in sublist]

        m, b, r_value, p_value, std_err = stats.linregress(independent,dependent)
        ri = analysis.dataRange([variables[0]], self.data)
        rd = analysis.dataRange([variables[1]], self.data)

        xmin, xmax = ri[0][0], ri[0][1]
        ymin, ymax = rd[0][0], rd[0][1]

        y0 = ((xmin * m + b) - ymin)/(ymax - ymin)
        y1 = ((xmax * m + b) - ymin)/(ymax - ymin)

        self.regEndpoints = np.matrix([[0,1],
                                       [y0, y1],
                                       [0,0],
                                       [1,1]])
        tend = vtm * self.regEndpoints
        print tend
        self.regLines.append(self.canvas.create_line(tend[0,0], tend[1,0],
                                                     tend[0,1], tend[1,1], fill="blue"))

        output = ("Regression slope: " , "%.3f" % float(m) ,
         "  Intercept: " , "%.3f" % float(b) , "  R-Value: " , "%.3f" % float(r_value * r_value))
        print output
        self.regLabel.config(text=output)

    #Resets view to standard xy plane
    def resetView(self):
        self.view = View()
        self.updateAxes()
        if self.spacePoints != None:
            self.updatePoints()

    def buildMenus(self):

        # create a new menu
        menu = tk.Menu(self.root)

        # set the root menu to our new menu
        self.root.config(menu = menu)

        # create a variable to hold the individual menus
        menulist = []

        # create a file menu
        filemenu = tk.Menu( menu )
        menu.add_cascade( label = "File", menu = filemenu )
        menulist.append(filemenu)

        # create another menu for kicks
        cmdmenu = tk.Menu( menu )
        menu.add_cascade( label = "Command", menu = cmdmenu )
        menulist.append(cmdmenu)

        anamenu = tk.Menu( menu )
        menu.add_cascade( label = "Analysis", menu = anamenu )
        menulist.append(anamenu)

        # menu text for the elements
        # the first sublist is the set of items for the file menu
        # the second sublist is the set of items for the option menu
        menutext = [ [ 'Open File', 'Clear \xE2\x8C\x98-N', 'Quit \xE2\x8C\x98-Q' ],
                     [ 'Reset View', 'Capture View' ],
                     ['Linear Regression', 'PCA', 'K-Means Clustering'] ]

        # menu callback functions (note that some are left blank,
        # so that you can add functions there if you want).
        # the first sublist is the set of callback functions for the file menu
        # the second sublist is the set of callback functions for the option menu
        menucmd = [ [self.handleOpen, self.clearData, self.handleQuit],
                    [self.resetView, self.captureView],
                    [self.handleLinearRegression, self.handlePCA, self.handleCluster] ]

        # build the menu elements and callbacks
        for i in range( len( menulist ) ):
            for j in range( len( menutext[i]) ):
                if menutext[i][j] != '-':
                    menulist[i].add_command( label = menutext[i][j], command=menucmd[i][j] )
                else:
                    menulist[i].add_separator()

    # create the canvas object
    def buildCanvas(self):
        self.canvas = tk.Canvas( self.root, width=self.initDx, height=self.initDy )
        self.canvas.pack( expand=tk.YES, fill=tk.BOTH )
        return

    # build a frame and put controls in it
    def buildControls(self):

        ### Control ###
        # make a control frame on the right
        self.rightcntlframe = tk.Frame(self.root)
        self.rightcntlframe.pack(side=tk.RIGHT, padx=2, pady=2, fill=tk.Y)

        # make a separator frame
        self.sep = tk.Frame( self.root, height=self.initDy, width=2, bd=1, relief=tk.SUNKEN )
        self.sep.pack( side=tk.RIGHT, padx = 2, pady = 2, fill=tk.Y)

        # use a label to set the size of the right panel
        label = tk.Label( self.rightcntlframe, text="Control Panel", width=20 )
        label.pack( side=tk.TOP, pady=10 )

        plotButton = tk.Button(self.rightcntlframe, text="Plot Data",
                                command=self.handlePlotData)
        plotButton.pack(side=tk.TOP)

        self.statusBar = tk.Frame(self.root, bg='white')
        self.statusBar.pack(side=tk.BOTTOM, padx=2, pady=2, fill=tk.Y)

        self.statusLabel = tk.Label( self.statusBar, text="Hello", height=1)
        self.statusLabel.pack( side=tk.TOP, pady=0 )

        self.regLabel = tk.Label( self.statusBar, text="  ", height=1)
        self.regLabel.pack( side=tk.BOTTOM, pady=0 )

        delPCAButton = tk.Button(self.rightcntlframe, text="Delete Analysis",
                                    command=self.handlePCADel)
        delPCAButton.pack(side=tk.BOTTOM, pady = 0)

        dispPCAButton = tk.Button(self.rightcntlframe, text="Display Info",
                                    command=self.handlePCAinfo)
        dispPCAButton.pack(side=tk.BOTTOM, pady = 0)

        plotPCAButton = tk.Button(self.rightcntlframe, text="Plot Analysis",
                                    command=self.handlePCAPlot)
        plotPCAButton.pack(side=tk.BOTTOM, pady = 0)

        self.pcaBox = tk.Listbox(self.rightcntlframe, selectmode=tk.BROWSE)
        self.pcaBox.pack(side=tk.BOTTOM, pady = 5)
        b = tk.Label(self.rightcntlframe, text="PCA Analyses", width=20)
        b.pack(side = tk.BOTTOM, pady=0)

    def setBindings(self):
        # bind mouse motions to the canvas
        self.canvas.bind( '<Motion>', self.motion)
        self.canvas.bind( '<Button-1>', self.handleMouseButton1 )
        self.canvas.bind( '<Control-Button-1>', self.handleMouseButton2 )
        self.canvas.bind( '<Command-Button-1>', self.handleMouseButton3)
        self.canvas.bind( '<Command-B1-Motion>', self.handleMouseButton3Motion)
        self.canvas.bind( '<Button-2>', self.handleMouseButton2 )
        self.canvas.bind( '<B1-Motion>', self.handleMouseButton1Motion )
        self.canvas.bind( '<B2-Motion>', self.handleMouseButton2Motion )
        self.canvas.bind( '<Control-B1-Motion>', self.handleMouseButton2Motion )

        self.root.bind( '<Command-q>', self.handleQuit )
        self.root.bind( '<Command-o>', self.handleOpen )
        self.root.bind( '<Command-n>', self.clearData ) # To clear data

    def handleQuit(self, event=None):
        print 'Terminating'
        self.root.destroy()

    def handleOpen(self, event=None):
        fn = tkFileDialog.askopenfilename(parent=self.root,
                                          title='Choose a data file',
                                          initialdir='.')
        if(fn != '' and ".csv" in fn):
            self.data = Data(fn)
            try:
                self.buildOptions()
                print("Success")
            except AttributeError:
                print("No Data yet!")

            return


    def buildOptions(self):
        try:
            self.xlabel.pack_forget()
            self.xDrop.pack_forget()
            self.ylabel.pack_forget()
            self.yDrop.pack_forget()
            self.zlabel.pack_forget()
            self.zDrop.pack_forget()
            self.slabel.pack_forget()
            self.sizeDrop.pack_forget()
            self.clabel.pack_forget()
            self.colorDrop.pack_forget()
            self.cbutton.pack_forget()
        except:
            print("Whoops")


        self.xAxisOp = tk.StringVar(self.root)

        self.yAxisOp = tk.StringVar(self.root)

        self.zAxisOp = tk.StringVar(self.root)

        self.sizeOp  = tk.StringVar(self.root)

        self.colorOp = tk.StringVar(self.root)

        options = self.data.headersNumeric
        optionals = ["Ignore"] + options

        self.xAxisOp.set(options[0])
        self.yAxisOp.set(options[0])
        self.zAxisOp.set(optionals[0])
        self.sizeOp.set(optionals[0])
        self.colorOp.set(optionals[0])

        self.xDrop = apply(tk.OptionMenu, (self.rightcntlframe, self.xAxisOp) + tuple(options))
        self.yDrop = apply(tk.OptionMenu, (self.rightcntlframe, self.yAxisOp) + tuple(options))
        self.zDrop = apply(tk.OptionMenu, (self.rightcntlframe, self.zAxisOp) + tuple(optionals))
        self.sizeDrop = apply(tk.OptionMenu, (self.rightcntlframe, self.sizeOp) + tuple(optionals))
        self.colorDrop = apply(tk.OptionMenu, (self.rightcntlframe, self.colorOp) + tuple(optionals))

        self.xlabel = tk.Label( self.rightcntlframe, text="X-Axis Data", width=20 )
        self.xlabel.pack( side=tk.TOP, pady=5 )
        self.xDrop.pack(side=tk.TOP)
        self.ylabel = tk.Label( self.rightcntlframe, text="Y-Axis Data", width=20 )
        self.ylabel.pack( side=tk.TOP, pady=5 )
        self.yDrop.pack(side=tk.TOP)
        self.zlabel = tk.Label( self.rightcntlframe, text="Z-Axis Data", width=20 )
        self.zlabel.pack( side=tk.TOP, pady=5 )
        self.zDrop.pack(side=tk.TOP)
        self.slabel = tk.Label( self.rightcntlframe, text="Size Data", width=20 )
        self.slabel.pack( side=tk.TOP, pady=5 )
        self.sizeDrop.pack(side=tk.TOP)
        self.clabel = tk.Label( self.rightcntlframe, text="Color Data", width=20 )
        self.clabel.pack( side=tk.TOP, pady=5 )
        self.colorDrop.pack(side=tk.TOP)
        self.cbutton = tk.Button(self.rightcntlframe, text="Discrete Colors",
                                    command=self.colorFix)
        self.cbutton.pack(side = tk.TOP)


    def handlePlotData(self, event=None):
        if self.data == None:
            self.handleOpen()
        else:
            self.color = None
            self.size = None
            self.buildPoints(self.handleChooseAxes())

    def handleChooseAxes(self):
        headersForPoints = []
        headersForPoints.append(self.xAxisOp.get())
        headersForPoints.append(self.yAxisOp.get())
        headersForPoints.append(self.zAxisOp.get())
        headersForPoints.append(self.sizeOp.get())
        headersForPoints.append(self.colorOp.get())
        return headersForPoints

    def handleCluster(self):
        d = clusterDialog(self.root, self.data.getHeaders())
        if d.result != None and int(d.clusters) > 0:
            print d.result
            headers = []
            for index in d.result:
                headers.append(self.data.headersNumeric[index])
            codebook, codes, errors = analysis.kmeans(self.data, headers, int(d.clusters))
            print codes
            self.clusterCount += 1
            print self.clusterCount
            self.data.addColumn("Clusters %d" % (self.clusterCount,), codes)
            self.buildOptions()

    def handlePCA(self):
        if self.data == None:
            self.handleOpen()
        else:
            self.pcacount += 1
            d = ColDialog(self.root, self.data.getHeaders())
            if len(d.result) > 0:
                w = NameDialog(self.root, self.pcacount)
                if w.result != None:
                    headers = []
                    for i in d.result[0]:
                        headers.append(self.data.headersNumeric[i])
                        print self.data.headersNumeric[i]
                    self.pcaList.append(analysis.pca(self.data, headers, normalize = d.result[1]))
                    self.pcaBox.insert(tk.END, w.result)
                    self.pcaList[len(self.pcaList)-1].toFile()

    def handlePCAPlot(self):
        if len(self.pcaList) > 0:
            index = self.pcaBox.index(tk.ACTIVE)
            print(self.pcaList[index].get_eigenvectors())
            print(self.pcaList[index].get_eigenvalues())
            self.buildPCA(self.pcaList[index].getHeaders(), self.pcaList[index])

    def handlePCADel(self):
        index = self.pcaBox.index(tk.ACTIVE)
        self.pcaBox.delete(index)
        if len(self.pcaList > 0):
            del self.pcaList[index]

    def handlePCAinfo(self):
        if len(self.pcaList) > 0:
            index = self.pcaBox.index(tk.ACTIVE)
            vecs = self.pcaList[index].get_eigenvectors().tolist()
            vals = self.pcaList[index].get_eigenvalues().tolist()[0]
            rheaders = self.pcaList[index].getRawHeaders()
            headers = copy.copy(self.pcaList[index].getHeaders())
            headers.insert(0,"E-Val")
            headers.insert(0,"E-Vec")
            array = []
            array.append(headers)
            for rheader,val,vec in zip(rheaders,vals,vecs):
                vec.insert(0,val)
                vec.insert(0,rheader)
                array.append(vec)
            e = EigenDialog(self.root, array)
    #Nice little clearData method. It does what it is supposed to. Good method!
    def clearData(self, event=None):
        print 'clearing data'
        self.objects = []
        self.axes = []
        self.canvas.delete("all");
        self.buildAxes()

    def captureView(self, event=None):
        f = tkFileDialog.asksaveasfile(mode='w', defaultextension=".eps")
        if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
            return
        text2save = f.name
        self.canvas.postscript(file=text2save)

    def motion(self, event):
        x, y = event.x, event.y
        for idx, point in enumerate(self.objects):
            coords = self.canvas.coords(point)
            if coords[0]<= x and x <= coords[2] and coords[1]<= y and y <= coords[3]:
                output = ""
                for header in self.handleChooseAxes():
                    if header != "Ignore":
                        output += header + ": "
                        output += self.data.getRawValue(header, idx)
                        output += "  "
                self.statusLabel.config(text=output)
                break
            else:
                output = ""
                self.statusLabel.config(text=output)

    def handleButton1(self):
        print 'handling command button:', self.colorOption.get()
        for obj in self.objects:
            self.canvas.itemconfig(obj, fill="black" )

    def handleLinearRegression(self):
        d = LinRegDialog(self.root, self.data)

        if(d.result != None):
            self.clearData()
            self.regLines = []
            self.resetView()
            self.updateAxes()
            #print(d.result)
            self.buildLinearRegression(d.result)

    def handleMouseButton1(self, event):
        print 'handle mouse button 1: %d %d' % (event.x, event.y)
        self.baseClick = (event.x, event.y)
        print(self.view.vrp)

    def handleMouseButton2(self, event):
        self.baseClick = (event.x, event.y)
        self.ogExtent = self.view.extent.copy()
        print 'handle mouse button 2: %d %d' % (event.x, event.y)

    # This is called if the first mouse button is being moved
    def handleMouseButton1Motion(self, event):
        # calculate the difference
        diff = ( event.x - self.baseClick[0], event.y - self.baseClick[1] )
        #print 'handle button1 motion %d %d' % (diff[0], diff[1])

        screendiff = (float(diff[0])/self.view.screen[0,0],
                      float(diff[1])/self.view.screen[0,1])
        multdiff   = (screendiff[0]*self.view.extent[0,0],
                      screendiff[1]*self.view.extent[0,1])
        self.view.vrp = self.view.vrp + (multdiff[0]*self.view.u) + (multdiff[1]*self.view.vup)
        print(self.view.vrp)
        self.updateAxes()
        self.updatePoints()

        self.baseClick = (event.x, event.y)

    # This is called if the second button of a real mouse has been pressed
    # and the mouse is moving. Or if the control key is held down while
    # a person moves their finger on the track pad.
    def handleMouseButton2Motion(self, event):
        print 'handle button 2 motion'
        diff = event.y - self.baseClick[1]
        if diff > 400:
            diff = 400
        elif diff < -400:
            diff = -400
        if diff < 0:
            scalefac = (((diff + 400) * 0.9)/400) + 0.1
        elif diff > 0:
            scalefac = (((diff)* 3.0)/400) + 1
        self.view.extent = self.ogExtent * scalefac
        self.updateAxes()
        self.updatePoints()

    def handleMouseButton3(self, event):
        self.baseClick = (event.x, event.y)
        self.ogView = self.view.clone()

    def handleMouseButton3Motion(self,event):
        diff = (event.x - self.baseClick[0], event.y - self.baseClick[1])
        dx = diff[0] * math.pi/400
        dy = diff[1] * math.pi/400
        self.view = self.ogView.clone()
        self.view.rotateVRC(-dx, dy)
        self.updateAxes()
        self.updatePoints()

        self.baseClick = (event.x, event.y)
        self.ogView = self.view.clone()

    def main(self):
        print 'Entering main loop'
        self.root.mainloop()


if __name__ == "__main__":
    dapp = DisplayApp(850, 720)
    dapp.main()
