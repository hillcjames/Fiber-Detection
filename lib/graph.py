import numpy as np
from math import pi, cos, sin, sqrt, atan
from skimage.draw import line_aa
from random import randint, random

from lib.node import Node

class Graph(dict):
    def __init__(self, distIm, points, maxDist, fiberW):
        ()
        
    
    def removeArc(self, angleList, angle, dist):
        # times 5 to lessen likelihood of cycles
        #arcAngle is half of the angle to be removed
        arcAngle = 5 * atan( self.invSqrRt2 / dist )
        lBnd = angle - arcAngle
        rBnd = angle + arcAngle
        if lBnd < 0:
            lBnd += 2*pi
        if rBnd > 2*pi:
            rBnd -= 2*pi
#         print("\t\t\t", (int)(lBnd*1000)/1000, (int)(rBnd*1000)/1000)

        if type(angleList) == dict:
            for e in angleList:
                if angleList[e] == -1:
                    continue
                if angleList[e] > lBnd and angleList[e] < lBnd + 2 * arcAngle:
                    angleList[e] = -1
                elif angleList[e] < rBnd and angleList[e] > rBnd - 2 * arcAngle:
                    angleList[e] = -1
        else:
            for i in range(0, len(angleList)):
                if angleList[i] == -1:
                    continue
                if angleList[i] > lBnd and angleList[i] < lBnd + 2 * arcAngle:
                    angleList[i] = -1
                elif angleList[i] < rBnd and angleList[i] > rBnd - 2 * arcAngle:
                    angleList[i] = -1
        
    
    
    def removeArcMatrix(self, angleMatrix, angle, dist):
        # times 5 to lessen likelihood of cycles
        #arcAngle is half of the angle to be removed
        arcAngle = 2 * atan( self.invSqrRt2 / dist )
        lBnd = angle - arcAngle
        rBnd = angle + arcAngle
        if lBnd < 0:
            lBnd += 2*pi
        if rBnd > 2*pi:
            rBnd -= 2*pi
#         print("\t\t\t", (int)(lBnd*1000)/1000, (int)(rBnd*1000)/1000)
        for i1 in range(int(dist), len(angleMatrix)):
            for i2 in range(0, len(angleMatrix[i1])):
                if angleMatrix[i1][i2] == -1:
                    continue
                if angleMatrix[i1][i2] > lBnd and angleMatrix[i1][i2] < lBnd + 2 * arcAngle:
                    angleMatrix[i1][i2] = -1
                elif angleMatrix[i1][i2] < rBnd and angleMatrix[i1][i2] > rBnd - 2 * arcAngle:
                    angleMatrix[i1][i2] = -1
            
    
    @staticmethod
    def toPointDict(l):
        d = {}
        for p in l:
            d[p] = Graph.Point(p)
        return d
    
    def getEndpoints(self):
        self.ends = {}
        for p in self:
            if len(self[p].links) == 1:
                self.ends[p] = self[p]

    
    def prune(self, endList = []):
        pToDelete = []
        pToCut = []
        nToEnds = []
        for p0 in self:
            nodePoint = self[p0]
            if len(nodePoint.links) < 1:
                if self.shouldRemove(nodePoint): # currently returns true
                    pToDelete.append(nodePoint.e)
                
            if (len(nodePoint.links) == 2) and self.inLine(nodePoint.e, nodePoint.links[0].e, nodePoint.links[1].e):
                pToCut.append(nodePoint.e)
    
            if (len(nodePoint.links) == 1
                    and nodePoint.e not in self.ends
                    and nodePoint.e not in endList):
                nToEnds.append(nodePoint)
                
        for p in pToDelete:
            self[p].visited = True
#             if p in self.ends:
#                 self.ends.remove(self[p])
            if p in self.ends:
                del self.ends[p]
#             if p in endList:    #NOT needed and introduces a bug, since an index to the list will no longer lead to the same value, while iterating over the list
#                 endList.remove(p)
            del self[p]
            
        for p in pToCut:
            if (len(self[p].links) == 2):
#                 print("....", p, len(self[p].links))
#                 self[p].printLinks(",,,,")
                Node.cutOut(self[p])
                del self[p]
        
        if len(endList) > 0:
#             string = "4 "
            for n in nToEnds:
#                 string += str(n.e) + " " + str(n.e not in pToDelete)
                # should I do this? Is there a situation where a node will,
                # after other nodes are unlinked around it, still have a single link and yet want to be deleted?
                if n.e not in pToDelete:
                    endList.append(n.e)
                    self.ends[n.e] = n
#             print(string)
            
    def shouldRemove(self, node):
        return True
    
    @staticmethod
    def getradialCircleArray(maxR):
        divs = int(maxR*2*pi)
        radiusList = np.arange(1.0 , maxR, 0.5)
        # each row in the matrix is a list of theta values, corresponding to a point
        mat = np.zeros((len(radiusList), divs))
        mat -= 1
        
        pointDict = {}
        for iR in range(0, len(radiusList)):
            r = radiusList[iR]
            i = 0
            for iT in range(0, divs):
                t = 2*pi/divs * iT
                p = (int((r * np.array([cos(t), sin(t)]))[0]), 
                     int((r * np.array([cos(t), sin(t)]))[1]) )
                if p not in pointDict:
                    mat[(iR, i)] = t
                    pointDict[p] = True
                    i += 1
        return radiusList, mat

    
    def drawGraph(self, im = 0):
        assert False, "Abstract parent function"
    
    def getNextPoint(self, node, prev, angleTol):
        ()
            
    
    def getAngle(self, p1, p2):
        return atan((p2[1] - p1[1])/(p2[0] - p1[0]+0.1))
    
    def findFibers(self):
        assert False, "unimplemented"
    
    @staticmethod
    def unlinkChain( l ):
        for i in range(0, len(l) - 1):
            p1 = l[i]
            p2 = l[i + 1]
            if Node.linked(p1, p2):
                Node.unlink(p1, p2)
    
    def save(self, fileName):
        file = open(fileName, mode='w')
        for p in self:
            string = " ".join(str(x) for x in p) + " " + " ".join(" ".join(str(x) for x in p0.p) for p0 in self[p].links) + "\n"
            print(string)
            file.write(string)
    
    @staticmethod
    def load(fileName):
        file = open(fileName, mode='r')
        file.read
        graph = Graph()
        print("here")
        for l in file:
            print("__")
            l = l[:-1].split(" ")
            for i in range(0, len(l)):
                l[i] = int(l[i])
            print(l)
            p = (l[0], l[1])
            l = l[2:]
            print(l)
            point = Graph.Point(p)
            for i in range(0, len(l), 2):
                point.links.append((l[i], l[i+1]))
                
        for p in graph:
            for i in range(0, len(graph[p].links)):
                if graph[p].links[i] in graph:
                    graph[p].links[i] = graph[graph[p].links[i]]
                else:
                    print(graph[p].links[i])
