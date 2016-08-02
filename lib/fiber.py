'''
Created on May 14, 2016

@author: james
'''
import numpy as np
from math import pi, cos, sin, sqrt, atan
from skimage.draw import line_aa
from PIL import Image, ImageDraw
from lib.helperFuncs import sqrDist


class Fiber:
    def __init__(self, points, fiberW):
        self.pnts = points
        self.pAvg = (np.array(self.pnts[0]) + np.array(self.pnts[-1]))/2
        self.w = fiberW
        self.calcLineProperties()
    
    def calcLineProperties(self):
        self.calcAngle()
        self.calcLength()
    
    def calcAngle(self):
        p0, p1 = self.pnts[0], self.pnts[len(self.pnts)-1]
        self.angle = atan((p1[1] - p0[1])/(p1[0] - p0[0]+0.02))
    
    def calcLength(self):
        self.length = 0
        for i in range(0, len(self.pnts)-1):
            self.length += sqrt(sqrDist(self.pnts[i], self.pnts[i+1]))
    
    def draw(self, im, c, offset = (0, 0)):
        draw = ImageDraw.Draw(im)
        for i in range(0, len(self.pnts)-1):
            p1 = (self.pnts[i][0] - offset[0], self.pnts[i][1] - offset[1])
            p2 = (self.pnts[i+1][0] - offset[0], self.pnts[i+1][1] - offset[1])
            draw.ellipse([(p1[0]-self.w/2+1, p1[1]-self.w/2+1),(p1[0]+self.w/2-1, p1[1]+self.w/2-1)], fill = c)
#             draw.ellipse([(p1[0]-offset[0]-self.w/2+1, p1[1]-offset[1]-self.w/2+1),(p1[0]-offset[0]+self.w/2-1, p1[1]-offset[1]+self.w/2-1)], fill = c)
            draw.line([p1,p2], width = (self.w*6)//5, fill = c)
        p0 = (self.pnts[len(self.pnts)-1][0] - offset[0], self.pnts[len(self.pnts)-1][1] - offset[1])
        draw.ellipse([(p0[0]-self.w/2+1, p0[1]-self.w/2+1),(p0[0]+self.w/2-1, p0[1]+self.w/2-1)], fill = c)
        
    def getEndPointsStr(self):
        return str(self.pnts[0][0]) + " " + str(self.pnts[0][1]) + " " + str(self.pnts[len(self.pnts)-1][0]) + " " + str(self.pnts[len(self.pnts)-1][1])
        
    def getEndPoints(self):
        return self.pnts[0], self.pnts[-1]
