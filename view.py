import numpy as np
import math

# View class
# Holds the current viewing parameters
# and can build a view transformation
# matrix [VTM] based on the parameters.
# Carl-Philip Majgaard
# CS 251
# Spring 2016

class View:

    def __init__(self):
        self.vrp    = np.matrix([0.5, 0.5, 1])
        self.vpn    = np.matrix([0, 0, -1])
        self.vup    = np.matrix([0, 1, 0])
        self.u      = np.matrix([-1, 0, 0])
        self.extent = np.matrix([1, 1, 1])
        self.screen = np.matrix([600, 600])
        self.offset = np.matrix([20, 20])

    def build(self):
        vtm = np.identity(4, float)
        t1  = np.matrix( [[1, 0, 0, -self.vrp[0, 0]],
                          [0, 1, 0, -self.vrp[0, 1]],
                          [0, 0, 1, -self.vrp[0, 2]],
                          [0, 0, 0, 1] ] )
        vtm = t1 * vtm

        tu   = np.cross(self.vup, self.vpn)
        tvup = np.cross(self.vpn, tu)
        tvpn = self.vpn.copy()

        tu   = self.normalize(tu)
        tvup = self.normalize(tvup)
        tvpn = self.normalize(tvpn)

        self.vpn = tvpn.copy()
        self.vup = tvup.copy()
        self.u   = tu.copy()

        r1 = np.matrix( [[  tu[0, 0],   tu[0, 1],   tu[0, 2], 0.0 ],
                        [ tvup[0, 0], tvup[0, 1], tvup[0, 2], 0.0 ],
                        [ tvpn[0, 0], tvpn[0, 1], tvpn[0, 2], 0.0 ],
                        [        0.0,        0.0,        0.0, 1.0 ]] )

        vtm = r1 * vtm

        t2  = np.matrix( [[1, 0, 0, 0.5 * self.extent[0,0]],
                          [0, 1, 0, 0.5 * self.extent[0,1]],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]] )
        vtm = t2 * vtm

        sx = (-self.screen[0,0])/self.extent[0,0]
        sy = (-self.screen[0,1])/self.extent[0,1]
        sz = (1.0 / self.extent[0,2])

        s1 = np.matrix( [[sx, 0, 0, 0],
                         [0, sy, 0, 0],
                         [0, 0, sz, 0],
                         [0, 0, 0,  1]] )

        vtm = s1 * vtm

        t3 = np.matrix( [[1, 0, 0, self.screen[0,0] + self.offset[0,0]],
                         [0, 1, 0, self.screen[0,1] + self.offset[0,1]],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]] )

        vtm = t3 * vtm

        return vtm

    def normalize(self, vector):
        length = np.linalg.norm(vector)
        vector = vector/length
        return vector

    def rotateVRC(self, vupa, ua):
        point = self.vrp + self.vpn * self.extent[0,2]*0.5
        t1 = np.matrix( [[1, 0, 0, -point[0,0]],
                          [0, 1, 0, -point[0,1]],
                          [0, 0, 1, -point[0,2]],
                          [0, 0, 0, 1]] )
        rxyz = np.matrix( [[self.u[0,0], self.u[0,1], self.u[0,2], 0],
                           [self.vup[0,0], self.vup[0,1], self.vup[0,2], 0],
                           [self.vpn[0,0], self.vpn[0,1], self.vpn[0,2], 0],
                           [0,0,0,1]])
        r1 = np.matrix( [[math.cos(vupa), 0, math.sin(vupa), 0],
                         [0,1,0,0],
                         [-math.sin(vupa), 0, math.cos(vupa), 0],
                         [0,0,0,1]])
        r2 = np.matrix( [[1,0,0,0],
                         [0,math.cos(ua),-math.sin(ua),0],
                         [0, math.sin(ua), math.cos(ua), 0],
                         [0,0,0,1]])

        t2 = np.matrix( [[1, 0, 0, point[0,0]],
                          [0, 1, 0, point[0,1]],
                          [0, 0, 1, point[0,2]],
                          [0, 0, 0, 1]] )

        tvrc = np.matrix( [[self.vrp[0,0], self.vrp[0,1], self.vrp[0,2], 1],
                           [self.u[0,0],self.u[0,1],self.u[0,2],0],
                           [self.vup[0,0],self.vup[0,1],self.vup[0,2],0],
                           [self.vpn[0,0],self.vpn[0,1],self.vpn[0,2],0]])

        tvrc = (t2*rxyz.T*r2*r1*rxyz*t1*tvrc.T).T

        self.vrp = tvrc[0, range(0,3)]
        self.u   = tvrc[1, range(0,3)]
        self.vup = tvrc[2, range(0,3)]
        self.vpn = tvrc[3, range(0,3)]

        self.vrp = self.vrp
        self.u = self.normalize(self.u)
        self.vup = self.normalize(self.vup)
        self.vpn = self.normalize(self.vpn)

    def clone(self):
        v        = View()
        v.vrp    = self.vrp.copy()
        v.vpn    = self.vpn.copy()
        v.vup    = self.vup.copy()
        v.u      = self.u.copy()
        v.extent = self.extent.copy()
        v.screen = self.screen.copy()
        v.offset = self.offset.copy()
        return v

if __name__ == "__main__":
    v = View()
    v.build()
