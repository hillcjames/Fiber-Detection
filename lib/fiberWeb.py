'''
Created on May 14, 2016

@author: james
'''
import numpy as np
from math import pi, cos, sin, sqrt

from lib.graph import Graph
from lib.fiber import Fiber
from lib.node import Node
from lib.helperFuncs import getAngle, sqrDist, getAngleDiff, getOrderedEndPoints

class FiberWeb(Graph):
    
    def __init__(self, imForSize, fibers, maxDist, fiberW):
        
        radiusList = np.arange(1.0 , maxDist, 0.4)
        thetaList0 = np.arange(0 , 2*pi, pi/60)

        for f in fibers:
            self[f] = Node(f)
            
        self.im = np.zeros((len(imForSize), len(imForSize[0])))
        self.fw = fiberW
        self.ends = {}
        self.invSqrRt2 = 1/sqrt(2)
        
        tempDict = {}
        for f in self:
            print(f.pnts[0], f.pnts[1])
            n = self[f]
            tempDict[f.pnts[0]] = n
            tempDict[f.pnts[1]] = n
        
        tolerance = pi/8
        
        i = 0
        for p0 in tempDict:
            print(p0)
            t0 = tempDict[p0].e.angle
#             pAvg = (np.array(tempDict[p0].e.pnts[0]) + np.array(tempDict[p0].e.pnts[-1]))/2
#             print(tempDict[p0].e.pnts[0], tempDict[p0].e.pnts[-1], pAvg)

#             if i%100 == 0:
#                 print(i, len(fibers), "building fiber graph")
            print(i, len(tempDict), "building fiber graph")
            i+=1
            
            thetaList = thetaList0.copy()
            for t in thetaList:
                if t == -1:
                    continue
                for r in radiusList:
                    p = (int(p0[0] + (r * np.array([cos(t), sin(t)]))[0]), 
                         int(p0[1] + (r * np.array([cos(t), sin(t)]))[1]) )
                    # if the point is an enpoint to a fiber, but not of the fiber currently being examined
                    
#                     if p in tempDict:
#                         print((int)(t*100)/100, r, p, p0, (p in tempDict), (p != tempDict[p0].pnts[0]), (p != tempDict[p0].pnts[1]))
                        
                    if (p in tempDict) and (p != tempDict[p0].e.pnts[0]) and (p != tempDict[p0].e.pnts[1]):
                        
#                         print("\t", (int)(t*100)/100, r, p, p0)
                        if not (Node.linked(tempDict[p0], tempDict[p])):
#                             print("\t\t", (int)(t*100)/100, r, p, p0)
                            
#                             if notCycle:
                            if self.aligned(tempDict[p0].e, tempDict[p].e, pi/8):
#                                 print("\t\t\t", (int)(t*100)/100, r, p, f0)
                                Node.link(tempDict[p0], tempDict[p])
                                self.removeArc(thetaList, t, r)
                                break
                        else:
                            break
        print()
        for f in self:
            print(str(f.pnts[0]) + " " + str(f.pnts[1]) + " " + str(self[f])[-7:] + "\t" + str(f)[-7:])
            for lnk in self[f].links: 
                print("\t", lnk.e.pnts[0], lnk.e.pnts[-1], "\t", str(lnk)[-7:], "\t", str(lnk.e)[-7:] )
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
        print("About to prune")
        self.prune()
        print("About to prune again")
        self.prune()
        print("...Done with prunes")
        
        self.getEndpoints()
    
    def shouldRemove(self, node):
#         print("should remove,", node.e.length, self.fw)
        return node.e.length < self.fw * 3
    
    ''' This finds and connects points into fibers in the pointWeb function:
    def findFibers(self):
        fiberList = []
        tempFiberList = []

        # print all nodes and their links, for debugging
        for e in self:
            print("*", e)
            temp = -1
            for n in self[e].links:
                print("\t*", n.e, n.e in self)
        
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
            print(prv.e)
            p0 = np.array(chain[0].e)
            while True:
                print(nxt.e, len(endList), [n.e for n in nxt.links])
                temp = -1
#                 for n in nxt.links:
#                     print("\t", n.e, sqrDist(nxt.e, n.e))
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
                        if len(nxt.links) == 1 and nxt.e not in self.ends:
                            self.ends[nxt.e] = nxt
                            endList.append(nxt.e)
#                         print(1)
                        break
                    
                    
                    if nxt.links[0] != prv:
                        temp = nxt.links[0]
                    else: 
                        temp = nxt.links[1]
                elif  len(nxt.links) == 1:
                    chain.append(nxt)
                    if nxt.e not in self.ends:
                        self.ends[nxt.e] = nxt
                        endList.append(nxt.e)
#                     print(2)
                    break
                else: #intersection
                    nxP = self.getNextPoint(nxt, chain[0], 5*pi/16)
                    print(nxP)
                    if nxP != 0:
                        temp = nxP
                        
                if temp == -1:
                    chain.append(nxt)
                    if len(nxt.links) == 1 and nxt.e not in self.ends:
                        self.ends[nxt.e] = nxt
                        endList.append(nxt.e)
#                     print(3)
                    break
                
                prv = nxt
                nxt = temp
#                 print(nxt)
                chain.append(nxt)
                
#                 if nxt.e not in self.ends:
#                     break
                
            chain0 = []
            for p in chain:
                chain0.append(p.e)
#             print(chain0)
            Graph.unlinkChain(chain)
            self.prune(endList)
            #don't prune the whole thing, just prune the points that had been touched
#             self.pruneChain()
            
            
            f = Fiber(chain0, self.fw)
            if f.length > 4:
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
    
    '''
    
    def findFibers(self):
        fiberList = []
        
#         print("fibers")
        for e in self.ends.copy():
#             print("e: ", e)
            if not e in self.ends or not e in self:
                continue
            if  len(self[e].links) == 0:
                fiberList.append(self[e].e)
                continue
                
            chain = []
            
            prv = self[e]
            nxt = self[e].links[0]
            chain.append(self[e])
            
            while True:
                temp = -1
#                 print(nxt.p, len(nxt.links))
                if len(nxt.links) == 2:
                    nxP = self.getNextPoint(nxt, prv, pi/8)
                    if nxP != 0:
                        if nxt.links[0] != prv:
                            temp = nxt.links[0]
                        else: 
                            temp = nxt.links[1]
                    else:
                        break
                elif  len(nxt.links) == 1:
                    chain.append(nxt)
                    break
                else: #intersection
                    nxP = self.getNextPoint(nxt, prv, pi/8)
                    if nxP != 0:
                        temp = nxP
                        
                if temp == -1:
                    chain.append(nxt)
                    break
                
                prv = nxt
                nxt = temp
#                 print(nxt)
                chain.append(nxt)
                
                if nxt.e not in self.ends:
                    break
                
            chain0 = []
            for nF in chain:
                chain0.append(nF.e)
#             print(chain0)
            Graph.unlinkChain(chain)
            self.prune()
            #don't prune the whole thing, just prune the points that had been touched
#             self.pruneChain()
            
            fPoints = getOrderedEndPoints(chain0[0], chain0[-1])[2:4]
            print(fPoints)
            f = Fiber(fPoints, self.fw)
            if f.length > 4:
                fiberList.append(f)
        return fiberList
    
    def getNextPoint(self, node, angleTol):
        minDist = 1000000
        # angleTol is the tolerance on either side. So half of total range.
        t0 = getAngle(node.e.pAvg, original.e.pAvg)
        # holds the points which lie nearly directly ahead
        aheadPoints = []
        for x in node.links:
            if x.e == original.e:
                continue
            t = getAngle(node.e.pAvg, x.e.pAvg)
            err = t - t0
            while err > pi:
                err -= 2 * pi 
            while err < -pi:
                err += 2 * pi
            dist = sqrDist(node.e.pAvg, x.e.pAvg)
            if abs(err) < angleTol and dist > dist0:
                aheadPoints.append( (x, abs(err), dist) )
        if len(aheadPoints) == 0:
            return 0
        aheadPoints = sorted(aheadPoints, key=lambda tup: tup[2])#, reverse=True)
        print(aheadPoints)
        return aheadPoints[0][0]

    def inLine(self, f0, f1, f2):
        tol = pi/8
        return self.aligned(f0, f1, tol) and self.aligned(f0, f2, tol)
    
    def aligned(self, f0, f1, tol):
        
        p1, p2 = f1.getEndPoints()
        
        t1 = getAngle(f0.pAvg, p1)
        t2 = getAngle(f0.pAvg, p2)
        return ((abs(getAngleDiff(f0.angle, t1)) <= tol) and (abs(getAngleDiff(f0.angle, t2)) <= tol))
    
    @staticmethod
    def fixBrokenFibers(fiberList, maxSeperation, angleTol):
        print("Hitching together broken fibers")
        i1 = 0
        fiberW = fiberList[0].w
        while(i1 < len(fiberList)):
            f1 = fiberList[i1]
            i2 = i1+1
            while(i2 < len(fiberList)):
                f2 = fiberList[i2]
                
                dist, newLength, far1, far2, near1, near2 = getOrderedEndPoints(f1, f2)
    #             if near1 == (87, 119) and near2 == (89, 114):
    #                 print(dist, abs(f1.angle - f2.angle), pi/24)
    #                 print(1/0)
                    
                endpointsNear = (dist < maxSeperation)
                sameSlope = abs(f1.angle - f2.angle) < angleTol
                avgAngle = (f1.angle + f2.angle)/2
    
                f3 = Fiber([far1, far2], fiberW)
                f4 = Fiber([near1, near2], fiberW)
                aligned = abs(avgAngle - f3.angle) < angleTol
                aligned = aligned and (abs(avgAngle - f4.angle) < angleTol)
                
                if endpointsNear and sameSlope and aligned and (newLength > f1.length) and (newLength > f2.length):
                    fiberList.remove(f2)
    #                 f1 = Fiber([far1, far2], fiberW)
                    f1 = f3
                    fiberList[i1] = f1
                    i2 -= 1
                i2 += 1
            i1 += 1
        return fiberList


