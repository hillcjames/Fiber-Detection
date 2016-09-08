'''
Created on May 14, 2016

@author: james
'''
import numpy as np
from math import pi, sqrt, sin, cos
from skimage.draw import line_aa
from random import randint

from lib.graph import Graph
from lib.fiber import Fiber
from lib.node import Node
from lib.helperFuncs import getAngle, getAngleDiff, sqrDist, intTup

class PointWeb(Graph):
    
    def __init__(self, distIm, points, maxDist, fiberW):
        # im0 is a binary image where all points which should be connected are connected
        # by a straight line of white pixels.
        # for each point, look in a circle until you hit another point or reach the maximum
        # link distance. 
        radiusList = np.arange(1.0 , maxDist, 0.4)
        thetaList0 = np.arange(0 , 2*pi, pi/60)

        for p in points:
            self[p] = Node(p)
            
        self.im = np.zeros((len(distIm), len(distIm[0])))
        self.fw = fiberW
        self.ends = {}
        self.invSqrRt2 = 1/sqrt(2)
    
        i = 0
        for p0 in self:
            if i%100 == 0:
                print(i, len(points), "building point graph")
            i+=1
            
            thetaList = thetaList0.copy()
            for r in radiusList:
                for t in thetaList:
                    if t == -1:
                        continue
                    p = (int(p0[0] + (r * np.array([cos(t), sin(t)]))[0]), 
                         int(p0[1] + (r * np.array([cos(t), sin(t)]))[1]) )
                    if (p != p0) and (p in self):
#                         if p == (46, 127) or p0 == (46, 127) or p == (47, 130) or p0 == (47, 134) or p == (50, 130) or p0 == (50, 134):
#                             print(p0, p)
#             treeName = spatial.cKDTree(points, leafsize=40)
#             for item in points:
#                         indx = treeName.query(item, distance_upper_bound=maxDist)[1]
                        
#                         print("\t", (int)(t*100)/100, r, p, p0)
                        if not (Node.linked(self[p], self[p0])):
#                             print("\t\t", (int)(t*100)/100, r, p, p0)
                            notCycle = True
                              
                            for n1 in self[p0].links:
                                p1 = n1.e
                                # if the points are adjacent, l/r/u/d.
                                if abs(p[0] - p1[0]) + abs(p[1] - p1[1]) == 1:
                                    notCycle = False
                                
                            for n2 in self[p].links:
                                # if the new point is connected to a node already linked to the current node
                                # prevents cycles of order 3
                                if Node.linked(self[p0], n2):
                                    notCycle = False
                                    break
                            
#                             if findPath(distIm, p0, p) and notCycle:
                            if notCycle:
#                                 print("\t\t\t", (int)(t*100)/100, r, p, p0)
                                Node.link(self[p], self[p0])
                                self.removeArc(thetaList, t, r)
                                break
                        else:
                            break
            
        # you need two, since I broke small cycles in the init,
        # so what would have been
        #     1-{2}, 2-{1, 3, 4}, 3-{2, 4}, 4-{2, 3, 5}, 5-{4)
        # is now
        #     1-{2}, 2-{1, 3}, 3-{2, 4}, 4-{2, 3, 5}, 5-{4)
        # which yields after the first prune
        #     1-{2}, 2-{1, 3}, 3-{2, 4}, 4-{3, 5}, 5-{4)
        # which yields
        #     1-{2}, 2-{1, 3}, 3-{2, 4}, 4-{3, 5}, 5-{4), then
        #     1-{2}, 2-{1, 3}, 4-{2, 5}, 5-{4), then
        #     1-{5}, 5-{1)
        
        self.getEndpoints()
        
        toDelete = []
         
        for e in self.ends:
            if len(self[e].links[0].links) > 2:
                toDelete.append(e)
         
        for e in toDelete:
            Node.unlink(self[e].links[0], self[e])
            del self[e]
            del self.ends[e]
        
        print("About to prune")
        ''' this does the basic prune, but also breaks some 3-node cycles '''
        self.prune()
        print("About to prune again")
        ''' this removes the unnecessary pieces from those broken cycles '''
        self.prune()
        
        ''' combine adjacent nodes'''
#         for e in self:
#             print("@@", e)
#             temp = -1
#             for n in self[e].links:
#                 print("\t@@", n.e, n.e in self)

        for curPoint in self.copy():
            if not curPoint in self:
                continue
            for linkedNode in self[curPoint].links:
                # if the manhatten distance <= 2; if they're touching
                if abs(curPoint[0] - linkedNode.e[0]) + abs(curPoint[1] - linkedNode.e[1]) <= 2:
                    # link non-duplicate links in e2 to e1, unlink everything from e2, then delete e2
                    for l in linkedNode.links:
                        if (l not in self[curPoint].links) and (l.e != curPoint):
                            Node.link(self[curPoint], l)
                    
                    # Bug fixed:
                    # Links existed to nodes that didn't, changing this to use a copy() fixed it.
                    # Removing things from a list while iterating over it isn't good.
                    for l in linkedNode.links.copy():
                        Node.unlink(linkedNode, l) # this includes curPoint
                        
                    if len(linkedNode.links) > 0:
                        raise Exception(len(linkedNode.links))
                    if linkedNode.e in self:
                        del self[linkedNode.e]
                    if linkedNode.e in self.ends:
                        del self.ends[linkedNode.e]
        
        print("Last prune")
        ''' this prunes again after adjacent nodes are combined '''
        self.prune()
        print("...Done with prunes")
        
        self.getEndpoints()

    def findFibers(self):
        fiberList = []
        tempFiberList = []
        
        '''
        print all nodes and their links, for debugging
        '''
#         for e in self:
#             print("*", e)
#             temp = -1
#             for n in self[e].links:
#                 print("\t*", n.e, n.e in self)
        
        endList = list(self.ends.copy())
        i = 0
        while i < len(endList):
            e = endList[i]
            i+=1
#             print("e: ", e)
            if not e in self.ends or not e in self:
                continue
            
            chain = []
            
            prv = self[e]
            nxt = self[e].links[0]
            chain.append(self[e])
#             print("@",prv.e)
            p0 = np.array(chain[0].e)
            while True:
                print("*", nxt.e, len(endList), [n.e for n in nxt.links])
                temp = -1
#                 for n in nxt.links:
#                     print("\t*", n.e, sqrDist(nxt.e, n.e))
                if len(nxt.links) == 2:
                    # this ends the fiber if it has too high a bend, and the points being examined
                    # aren't too close.
                    prvP = np.array(prv.e)
                    nxtP = np.array(nxt.e)
#                     print(p0 - prvP, p0 - nxtP, p0, prvP, nxtP, getAngle(p0 - prvP, nxtP - prvP), sqrDist(chain[0].e, nxt.e), 3*self.fw)
                    if ((sqrDist(chain[0].e, prv.e) > 3*self.fw or chain[0].e == prv.e)
                        and sqrDist(chain[0].e, nxt.e) > 3*self.fw
                        and getAngle(prvP - p0, nxtP - p0) < 6*pi/8):
#                         and getAngle(p0 - prvP, nxtP - prvP) < 6*pi/8):
                        chain.append(nxt)
                        nxt.visited = True
                        if len(nxt.links) == 1 and nxt.e not in self.ends:
                            self.ends[nxt.e] = nxt
                            endList.append(nxt.e)
#                             print("1*")
                        break
                    
                    
                    if nxt.links[0] != prv:
                        temp = nxt.links[0]
                    else:
                        temp = nxt.links[1]
                    
                    if temp.visited:
                        # TODO maybe condense small cycles into a single point
                        # instead of just not letting it connect
                        Node.unlink(temp, nxt)
                        break
                        smallCycleLength = min([6, len(chain)])
                        
                        cycle = []
                        cycleFound = False
                        #find cycle
                        for i0 in range(smallCycleLength - 1, 0, -1):
                            n = chain[i0]
                            if n != temp:
                                cycle.append(n)
                            else:
                                cycleFound = True
                                break
                        #if cycle is small, remove those nodes from:
                        #chain, ends, and graph; everything.
                        # save all of their links that aren't to each other,
                        # and add a new node at their center with all their links
                        if cycleFound:
                            self.unlinkChain(cycle)
                            links = []
                            pAvg = np.array((0,0))
                            for n in cycle:
                                pAvg += np.array(n.e)
                                for l in n.links.copy():
                                    if l not in links:
                                        links.append(l)
                                    Node.unlink(n, l)
                                    
                            pAvg /= len(cycle)
                            newNode = Node(intTup(pAvg))
                            for l in links:
                                Node.link(newNode, l)
                        break
                    
                elif  len(nxt.links) == 1:
                    chain.append(nxt)
                    if nxt.e not in self.ends:
                        self.ends[nxt.e] = nxt
                        endList.append(nxt.e)
#                         print("2*")
                    break
                else: #intersection
                    nxP = self.getNextPoint(nxt, chain[0], 5*pi/16)
                    if nxP != 0:
#                         print("\t#", nxP.e)
                        temp = nxP
                        
                if temp == -1:
                    chain.append(nxt)
                    if len(nxt.links) == 1 and nxt.e not in self.ends:
                        self.ends[nxt.e] = nxt
                        endList.append(nxt.e)
#                         print("3*")
                    break
                if prv == nxt:
                    print("previous point same as next, at:", prv)
                    break
                
                prv = nxt
                nxt = temp
#                 print(nxt)
                chain.append(nxt)
                nxt.visited = True
                
#                 if nxt.e not in self.ends:
#                     break
                
            chain0 = []
            for p in chain:
                chain0.append(p.e)
                p.visited = False
#             print(chain0)
            Graph.unlinkChain(chain)
            self.prune(endList)
            #don't prune the whole thing, just prune the points that had been touched
#             self.pruneChain()
            
            
            f = Fiber(chain0, self.fw)
            if f.length > 3:
                tempFiberList.append(f)
        
        # this ensures duplicate fibers aren't saved
        tempDict = {}
        for f in tempFiberList:
#             print( f.pnts[0], f.pnts[-1] )
            name = ""
            if f.pnts[0][0] < f.pnts[-1][0]:
                orderGood = True
            elif f.pnts[0][0] == f.pnts[-1][0]:
                if f.pnts[0][1] < f.pnts[-1][1]:
                    orderGood = True
                else:
                    orderGood = False
            else:
                orderGood = False
            
            if orderGood:
                name = str(f.pnts[0]) + " " + str(f.pnts[-1])
            else:
                name = str(f.pnts[-1]) + " " + str(f.pnts[0])
#             print(name)
            if not name in tempDict:
#                 print("\tHere")
                tempDict[name] = f
            
        for key in tempDict:
            fiberList.append(tempDict[key])
        
        return fiberList

    def getNextPoint0(self, node, original, angleTol):
        # angleTol is the tolerance on either side. So half of total range.
        t0 = self.getAngle(original.e, node.e)
        best = (0, angleTol)
        for x in node.links:
            if x == original:
                continue
            t = self.getAngle(node.e, x.e)
            err = t - t0
            while err > pi:
                err -= 2 * pi 
            while err < -pi:
                err += 2 * pi
             
            if abs(err) < best[1]:
                best = (x, abs(err))
          
        return best[0]

    def getNextPoint(self, node, original, angleTol):
#         print("\t", original.e, node.e)
        dist0 = sqrDist(node.e, original.e)
        # angleTol is the tolerance on either side. So half of total range.
        t0 = getAngle(original.e, node.e)
        # holds the points which lie nearly directly ahead
        aheadPoints = []
        for x in node.links:
            if x == original:
                continue
            t = getAngle(node.e, x.e)
            err = t - t0
            while err > pi:
                err -= 2 * pi 
            while err < -pi:
                err += 2 * pi
            dist = sqrDist(node.e, x.e)
            if abs(err) < angleTol and dist > dist0:
                aheadPoints.append( (x, abs(err), dist) )
        if len(aheadPoints) == 0:
            return 0
        aheadPoints = sorted(aheadPoints, key=lambda tup: tup[2])#, reverse=True)
#         print("\t--", aheadPoints[0][0].e, node.e)
        return aheadPoints[0][0]
    
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
            
    def inLine(self, p0, p1, p2):
        t1 = getAngle(p0, p1)
        t2 = getAngle(p0, p2)
        if abs(getAngleDiff(t1, t2)) <= pi/4:
            return True
        return False
        
        
if __name__ == "__main__":
    from main import cPythonStuff
    cPythonStuff()


    