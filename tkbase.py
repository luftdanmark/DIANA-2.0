# TK Skeleton
# Modified by Carl-Philip Majgaard
# CS 251
# Spring 2016

import Tkinter as tk
import tkFont as tkf
import math
import random
from view import *
from data import *


# create a class to build and manage the display
class DisplayApp:

    def __init__(self, width, height):

        # create a tk object, which is the root window
        self.root = tk.Tk()

        # width and height of the window
        self.initDx = width
        self.initDy = height
        self.dx = 2

        # set up the geometry for the window
        self.root.geometry( "%dx%d+50+30" % (self.initDx, self.initDy) )

        # set the title of the window
        self.root.title("DataWiz") # give it a sick name

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
        self.data = Data("data.csv") # will hold the raw data someday.
        self.points = self.data.getData(["A", "B", "C"])
        self.baseClick = None # used to keep track of mouse movement
        self.xDist = "Uniform" # used to keep track of x distribution for rand
        self.yDist = "Uniform" # used to keep track of y distribution for rand
        self.randPoints = 100 # keeps track of number of points to be added

        self.view = View()
        self.endpoints = np.matrix( [[0,1,0,0,0,0],
                                     [0,0,0,1,0,0],
                                     [0,0,0,0,0,1],
                                     [1,1,1,1,1,1]] )
        self.axes = []
        self.buildAxes()
        self.scalefactor = 1
        self.addPoints()

    def buildAxes(self):
        vtm = self.view.build()
        tend = vtm * self.endpoints
        self.axes.append(self.canvas.create_line(tend[0,0], tend[1,0],
                                                 tend[0,1], tend[1,1], fill='red'))
        self.axes.append(self.canvas.create_line(tend[0,2], tend[1,2],
                                                 tend[0,3], tend[1,3], fill='green'))
        self.axes.append(self.canvas.create_line(tend[0,4], tend[1,4],
                                                 tend[0,5], tend[1,5], fill='blue'))

    def updateAxes(self):
        vtm = self.view.build()
        tend = vtm * self.endpoints
        for idx, line in enumerate(self.axes):
            self.canvas.coords(line, tend[0,idx*2], tend[1,idx*2],
                                     tend[0,idx*2+1], tend[1,idx*2+1])
        tpoints = self.homogCoord * vtm.T
        dx = self.dx
        for i, point in enumerate(self.objects):
            self.canvas.coords(point, tpoints[i,0]-dx, tpoints[i,1]-dx, tpoints[i,0]+dx, tpoints[i,1]+dx)


    def addPoints(self):
        vtm = self.view.build()
        self.homogCoord = np.ones((self.points.shape[0], 4))
        self.homogCoord[:,:-1] = self.points
        print(self.homogCoord)
        tend = self.homogCoord * vtm.T
        for i in range(0,tend.shape[0]):
            dx = self.dx
            pt = self.canvas.create_oval( tend[i,0]-dx, tend[i,1]-dx, tend[i,0]+dx, tend[i,1]+dx,
                                        fill=self.colorOption.get(), outline='')
            self.objects.append(pt)
        print(tend)

    def resetView(self):
        self.view = View()
        self.updateAxes()

    def createRandomDataPoints( self, event=None ):
        for i in range(0,self.randPoints):
            if(self.xDist == "Gaussian"): # if gaussian is selected
                x = random.gauss() # do it
            else:
                x = random.random() # do a uniform
            if(self.yDist == "Gaussian"): # same for y
                y = random.gauss()
            else :
                y = random.random()
            z = random.random()

            coords = [x,y,z,1]

            self.homogCoord = np.vstack((self.homogCoord, coords))
            dx = self.dx
            pt = self.canvas.create_oval( x-dx, y-dx, x+dx, y+dx,
                                        fill=self.colorOption.get(), outline='')
            self.objects.append(pt)
            self.updateAxes()

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

        # menu text for the elements
        # the first sublist is the set of items for the file menu
        # the second sublist is the set of items for the option menu
        menutext = [ [ 'Choose Distribution', 'Clear \xE2\x8C\x98-N', 'Quit \xE2\x8C\x98-Q' ],
                     [ 'Command 1', 'Reset View', '-' ] ]

        # menu callback functions (note that some are left blank,
        # so that you can add functions there if you want).
        # the first sublist is the set of callback functions for the file menu
        # the second sublist is the set of callback functions for the option menu
        menucmd = [ [self.handleDist, self.clearData, self.handleQuit],
                    [self.handleMenuCmd1, self.resetView, None] ]

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
        rightcntlframe = tk.Frame(self.root)
        rightcntlframe.pack(side=tk.RIGHT, padx=2, pady=2, fill=tk.Y)

        # make a separator frame
        sep = tk.Frame( self.root, height=self.initDy, width=2, bd=1, relief=tk.SUNKEN )
        sep.pack( side=tk.RIGHT, padx = 2, pady = 2, fill=tk.Y)

        # use a label to set the size of the right panel
        label = tk.Label( rightcntlframe, text="Control Panel", width=20 )
        label.pack( side=tk.TOP, pady=10 )

        # make a menubutton
        self.colorOption = tk.StringVar( self.root )
        self.colorOption.set("black")
        colorMenu = tk.OptionMenu( rightcntlframe, self.colorOption,
                                        "black", "blue", "red", "green" ) # can add a command to the menu
        colorMenu.pack(side=tk.TOP)

        # make a button in the frame
        # and tell it to call the handleButton method when it is pressed.
        button = tk.Button( rightcntlframe, text="Update Color",
                               command=self.handleButton1 )
        button.pack(side=tk.TOP)  # default side is top

        randomButton = tk.Button( rightcntlframe, text="Add Random Points",
                                command=self.createRandomDataPoints)
        randomButton.pack(side=tk.TOP)

        return

    def setBindings(self):
        # bind mouse motions to the canvas
        self.canvas.bind( '<Button-1>', self.handleMouseButton1 )
        self.canvas.bind( '<Control-Button-1>', self.handleMouseButton2 )
        self.canvas.bind( '<Command-Button-1>', self.handleMouseButton3)
        self.canvas.bind( '<Command-B1-Motion>', self.handleMouseButton3Motion)
        self.canvas.bind( '<Shift-Command-Button-1>', self.handleShiftCommand )
        self.canvas.bind( '<Button-2>', self.handleMouseButton2 )
        self.canvas.bind( '<B1-Motion>', self.handleMouseButton1Motion )
        self.canvas.bind( '<B2-Motion>', self.handleMouseButton2Motion )
        self.canvas.bind( '<Button-3>', self.handleWheelClick )
        # bind command sequences to the root window
        self.canvas.bind( '<Control-B1-Motion>', self.handleMouseButton2Motion )

        self.root.bind( '<Command-q>', self.handleQuit )
        self.root.bind( '<Command-n>', self.clearData ) # To clear data

    def handleQuit(self, event=None):
        print 'Terminating'
        self.root.destroy()

    #Nice little clearData method. It does what it is supposed to. Good method!
    def clearData(self, event=None):
        print 'clearing data'
        for obj in self.objects:
            self.objects = []
            self.canvas.delete("all");

    def handleButton1(self):
        print 'handling command button:', self.colorOption.get()
        for obj in self.objects:
            self.canvas.itemconfig(obj, fill=self.colorOption.get() )

    def handleMenuCmd1(self):
        print 'handling menu command 1'

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

        self.baseClick = self.baseClick = (event.x, event.y)

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

    def handleMouseButton3(self, event):
        self.baseClick = (event.x, event.y)
        self.ogView = self.view.clone()

    def handleMouseButton3Motion(self,event):
        diff = (event.x - self.baseClick[0], event.y - self.baseClick[1])
        dx = diff[0] * math.pi/300
        dy = diff[1] * math.pi/300
        self.view = self.ogView.clone()
        self.view.rotateVRC(-dx, dy)
        self.updateAxes()


    # This method handles mb3 clicks. It drops a dot on the canvas where I click.
    def handleWheelClick(self, event):
        print 'handle wheel click'
        dx = 3
        rgb = "#%02x%02x%02x" % (random.randint(0, 255),
                             random.randint(0, 255),
                             random.randint(0, 255) )
        oval = self.canvas.create_oval( event.x - dx,
                                    event.y - dx,
                                    event.x + dx,
                                    event.y + dx,
                                    fill = rgb,
                                    outline='')
        self.objects.append( oval )

    #This method handles the distribution select process
    def handleDist(self):
        print 'handle choose Dist'
        #spawn dialog
        d = Dialog(self.root, self.xDist, self.yDist, self.randPoints)
        #if we've got something on the hook, assign our fields
        if(d.result != None):
            self.xDist = d.result[0]
            self.yDist = d.result[1]
            self.randPoints = int(d.result[2])

    #This method deletes items at a clicked coordinate
    def handleShiftCommand(self, event):
        dx = self.dx + 3
        items = self.canvas.find_enclosed(event.x - dx,
                                    event.y - dx,
                                    event.x + dx,
                                    event.y + dx)
        for item in items:
            self.canvas.delete(item)
            self.objects.remove(item)


    def main(self):
        print 'Entering main loop'
        self.root.mainloop()

#This is my dialog class.
class Dialog(tk.Toplevel):
    #Modified the init method to accept some args specifying current selections
    def __init__(self, parent, xDist, yDist, randPoints, title = 'Choose Distribution'):

        tk.Toplevel.__init__(self, parent)
        self.transient(parent)

        if title:
            self.title(title)

        self.parent = parent

        self.result = None

        body = tk.Frame(self)
        self.initial_focus = self.body(body, xDist, yDist, randPoints)
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
    # Modified arguments, now the also take current selections
    def body(self, master, xDist, yDist, randPoints):
        box = tk.Frame(self)

        w = tk.Label(master, text="X-AXIS")
        w.pack()

        #Make a listbox, and select the current choice
        self.l0 = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
        for item in ["Gaussian", "Uniform"]:
            self.l0.insert(tk.END, item)
        self.l0.config(height=5)
        self.l0.pack()

        if (xDist == "Gaussian"):
            self.l0.selection_set(0)
            self.l0.activate(0)
        else:
            self.l0.selection_set(1)
            self.l0.activate(1)

        w = tk.Label(master, text="Y-AXIS")
        w.pack()

        #Make a listbox, and select the current choice
        self.l1 = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
        for item in ["Gaussian", "Uniform"]:
            self.l1.insert(tk.END, item)
        self.l1.config(height=5)
        self.l1.pack()

        if (yDist == "Gaussian"):
            self.l1.selection_set(0)
            self.l1.activate(0)
        else:
            self.l1.selection_set(1)
            self.l1.activate(1)


        w = tk.Label(master, text="# of Points")
        w.pack()

        #make an entry and insert the current selection
        self.e = tk.Entry(master)
        self.e.pack()

        self.e.insert(0, randPoints)

        box.pack()

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

        self.cancel()

    def cancel(self, event=None):

        # put focus back to the parent window
        self.parent.focus_set()
        self.destroy()

    #
    # command hooks

    def validate(self):
        self.result = self.l0.get(tk.ACTIVE), self.l1.get(tk.ACTIVE), self.e.get()
        return 1

    def apply(self):

        pass # override


if __name__ == "__main__":
    dapp = DisplayApp(1200, 675)
    dapp.main()
