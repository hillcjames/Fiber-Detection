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
        arcAngle = 5 * atan( self.invSqrRt2 / dist )
        lBnd = angle - arcAngle
        rBnd = angle + arcAngle
        if lBnd < 0:
            lBnd += 2*pi
        if rBnd > 2*pi:
            rBnd -= 2*pi

        for i in range(0, len(angleList)):
            if angleList[i] == -1:
                continue
            if angleList[i] > lBnd and angleList[i] < lBnd + 2 * arcAngle:
                angleList[i] = -1
            elif angleList[i] < rBnd and angleList[i] > rBnd - 2 * arcAngle:
                angleList[i] = -1
            
    
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
    
    
    def drawGraph(self, im = 0):
        if im == 0:
            im = self.im
    #     for p in g:
    #         g[p].visited = False
        visited = {}
        for p1 in self:
            if p1 in visited:
                continue
            for nodeP in self[p1].links:
                visited[nodeP.e] = True
                p2 = nodeP.e
                rr, cc = line_aa(*p1 + p2)[:2]
                im[rr, cc] = 35  + randint(0, 40)
        for p in self:
            self.im[p] = 168
            
        for p in self.ends:
            self.im[p] = 255
    
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
