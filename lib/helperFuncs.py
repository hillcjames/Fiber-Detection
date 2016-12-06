'''
Created on May 29, 2016

@author: james
'''
from math import atan, atan2, sin, cos
from skimage.draw import line_aa

def sqrDist( p1, p2 ):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def intTup( t ):
    return (int(t[0]), int(t[1]))


def getAngle(p0, p2):
    return atan((p2[1] - p0[1])/(p2[0] - p0[0]+0.1))

def getOrderedEndPoints(f1, f2):
    # this returns 2 distances and 4 points, in order: 
    #   min, max, 2 farthest points, 2 nearest points (min, max, p3, p4)
    p1, p2 = f1.getEndPoints()
    p3, p4 = f2.getEndPoints()
    dist13 = sqrDist(p1, p3)
    dist14 = sqrDist(p1, p4)
    dist23 = sqrDist(p2, p3)
    dist24 = sqrDist(p2, p4)
    
    l = [(dist13, p1, p3), (dist14, p1, p4), (dist23, p2, p3), (dist24, p2, p4)]
    
    minDist = (100000000, 0)
    maxDist = (-5, 0)
    for i1 in range(0, 4):
        if l[i1][0] >= maxDist[0]:
            maxDist = l[i1]
        if l[i1][0] <= minDist[0]:
            minDist = l[i1]
    
    p2, p3 = minDist[1:]
    p1, p4 = maxDist[1:]
    return minDist[0], maxDist[0], p1, p4, p2, p3

def getConnectingPoints(p1, p2):
    rr, cc, val = line_aa(*p1 + p2)
    points = list(zip(rr, cc))
    
    return points

def getAngleDiff(x, y):
    return atan2(sin(x-y), cos(x-y))

