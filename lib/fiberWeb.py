'''
Created on May 14, 2016

@author: james
'''
import numpy as np
from math import pi, cos, sin, sqrt
from skimage.draw import line_aa
from random import randint
from scipy import spatial

from lib.graph import Graph
from lib.fiber import Fiber
from lib.node import Node
from lib.helperFuncs import getAngle, sqrDist, getAngleDiff, getOrderedEndPoints

class FiberWeb(Graph):
    
    '''
    TODO !!!!
    Simplify the graph by dividing all the points by 10 or 15 or something, and then use those
    smaller values to add things to the dictionary and perform the search.
    This will lose the difference between nodes within 10/15 pixels of each other. 
    Actually run the search on each node once on the actual data, from 0-10 or 15, and then run it
    again on the rounded data? Still loses data if any points are nearby to each other but not to the starting node
    Or not
    Also, add this somewhere
    gc.collect()
    '''
    
    
    def __init2__(self, imForSize, fibers, maxDist, fiberW):
        '''
        This orders points by x-coord, and then checks all points within maxDist of a point
        in the x-direction to see which of them are within maxDist in the y direction.
        These points are then narrowed down by a sqrDist calculation.
        '''

        for f in fibers:
            self[f] = Node(f)
            
        self.im = np.zeros((len(imForSize), len(imForSize[0])))
        self.fw = fiberW
        self.ends = {}
        self.invSqrRt2 = 1/sqrt(2)
        
        tempDict = {}
        for f in self:
            print(f.pnts[0], f.pnts[-1])
            n = self[f]
            tempDict[f.pnts[0]] = n
            tempDict[f.pnts[-1]] = n
        
        tolerance = pi/8
        
        dtype = [('x', int), ('y', int)]
        
        tempList = list(tempDict)
        
        pointList = np.sort(np.array(tempList, dtype), order='x')
        
        
        # Sort a list of all points
        
#         i = 0
        for i, p0 in enumerate(pointList):
#             pAvg = (np.array(tempDict[p0].e.pnts[0]) + np.array(tempDict[p0].e.pnts[-1]))/2
#             print(tempDict[p0].e.pnts[0], tempDict[p0].e.pnts[-1], pAvg)

            if i%100 == 0:
                print(i, len(fibers), "building fiber graph")
#             print(i, len(tempDict), "building fiber graph")
#             i+=1
            xLowBnd = int(p0[0] - maxDist)
            xHighBnd = int(p0[0] + maxDist)
            yLowBnd = int(p0[1] - maxDist)
            yHighBnd = int(p0[1] + maxDist)
            
            # this is an index
            lBnd = i
            while lBnd > 0 and pointList[lBnd - 1][0] >= xLowBnd:
                lBnd -= 1
                
#             lPnt = pointList[lBnd]
            
            rBnd = i
            while rBnd < len(pointList) - 2 and pointList[rBnd + 1][0] <= xHighBnd:
                rBnd += 1
                
#             rPnt = pointList[rBnd]
            
            for iX in range(lBnd, rBnd + 1):

                p = tuple(pointList[iX])
                if (yLowBnd < p[1] < yHighBnd):
                    p0 = tuple(p0)
                    if (i != iX) and (sqrDist(p, p0) <= maxDist**2) and (tempDict[p0] != tempDict[p]):
                         
#                     if (p in tempDict) and (p != tempDict[p0].e.pnts[0]) and (p != tempDict[p0].e.pnts[1]):
                        
#                         print("\t", (int)(t*100)/100, r, p, p0)
                        if not (Node.linked(tempDict[p0], tempDict[p])):
#                             print("\t\t", (int)(t*100)/100, r, p, p0)
#                             if notCycle:
                            notCycle = True
                            for n2 in tempDict[p].links:
                                # if the new point is connected to a node already linked to the current node
                                # prevents cycles of order 3
                                if Node.linked(tempDict[p0], n2):
                                    notCycle = False
                                    break
                            
                            if (self.aligned(tempDict[p0].e, tempDict[p].e, pi/32)
                                and getOrderedEndPoints(tempDict[p0].e, tempDict[p].e)[1] > tempDict[p0].e.length
                                and notCycle):
                                Node.link(tempDict[p0], tempDict[p])
#                                 self.removeArc(thetaMatrix[iR], t, r)
#                                 print("\t", (int)(t*1000)/1000, r, p0, p, "-", tempDict[p0].e.getEndPointsStr(), "-", tempDict[p].e.getEndPointsStr(), "-")
#                                 print(thetaList)
                                break
                        else:
#                             self.removeArc(thetaMatrix[iR], t, r)
#                             print("\t**", (int)(t*1000)/1000, r, p0, p, "-", tempDict[p0].e.getEndPointsStr(), "-", tempDict[p].e.getEndPointsStr(), "-")
#                             print("**",thetaList)
                            break
#         print()
#         for f in self:
#             print(str(f.pnts[0]) + " " + str(f.pnts[1]) + " " + str(self[f])[-7:] + "\t" + str(f)[-7:])
#             for lnk in self[f].links: 
#                 print("\t", lnk.e.pnts[0], lnk.e.pnts[-1], "\t", str(lnk)[-7:], "\t", str(lnk.e)[-7:] )
    
        for i, f in enumerate(self):
            string = " --- "
            if f in self.ends:
                string = " END "
            
            string = str(i) + string
            
            string += "|" + self[f].e.getEndPointsStr() + "| "
            for l in self[f].links:
                string += " - (" + l.e.getEndPointsStr() + ")"
            string += " -"
            
            print(string)
    
#         self.drawGraph(links=False)
#         temp = self.im.copy()
#         self.drawGraph(links=True)
#         from main import displayPlots
#         try:
#             displayPlots([temp, self.im])
#         except Exception:
#             ()
#         exit()
        
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
    
    def __init__(self, imForSize, fibers, maxDist, fiberW):
        '''
        This uses a PR-Quadtree to request points in a region.
        '''

        for f in fibers:
            self[f] = Node(f)
            
        self.im = np.zeros((len(imForSize), len(imForSize[0])))
        self.fw = fiberW
        self.ends = {}
        self.invSqrRt2 = 1/sqrt(2)
        
        thetaList0 = np.arange(0 , 2*pi, pi/300)

        tempDict = {}
        for f in self:
            print(f.pnts[0], f.pnts[-1])
            n = self[f]
            tempDict[f.pnts[0]] = n
            tempDict[f.pnts[-1]] = n
        
        
        pointList = np.array(list(tempDict))
        
        KDT = spatial.KDTree(pointList)

        
        # Sort a list of all points
        
#         i = 0
        for i, p0Np in enumerate(pointList):
            thetaList = thetaList0.copy()
            p0 = tuple(p0Np)
            
            
            if i%100 == 0:
                print(i, len(fibers), "building fiber graph")
#             print(i, len(tempDict), "building fiber graph")
            
            
            nearby = np.array(KDT.query_ball_point(p0, maxDist))
            nearArr = pointList[nearby] 
            
            distList = spatial.distance.cdist(np.array([p0]), nearArr)[0]
            nearPoints = nearArr[np.argsort(distList)]
            
            distList = np.sort(distList)
#             for iTemp, temp in enumerate(nearPoints):
#                 temp.append(distList[iTemp])
            
#             print(type(nearArr), type(nearArr[0]), type(p0Np))
#             
            for i2 in range(len(nearPoints)):
                pNp = nearPoints[i2]
                r = distList[i2]
                t = getAngle(p0, pNp)
                if t == -2:
                    break
#             for pIdx in nearPoints:
#                 p = pointList[pIdx]
                p = tuple(pNp)
#                 if p == 
                if (p != p0):
                    if tempDict[p0] != tempDict[p]:
                    
#                     if (p in tempDict) and (p != tempDict[p0].e.pnts[0]) and (p != tempDict[p0].e.pnts[1]):
                        
#                         print("\t", (int)(t*100)/100, r, p, p0)
                        if not (Node.linked(tempDict[p0], tempDict[p])):
#                             print("\t\t", (int)(t*100)/100, r, p, p0)
#                             if notCycle:
                            notCycle = True
                            for n2 in tempDict[p].links:
                                # if the new point is connected to a node already linked to the current node
                                # prevents cycles of order 3
                                if Node.linked(tempDict[p0], n2):
                                    notCycle = False
                                    break
                            
                            if (self.aligned(tempDict[p0].e, tempDict[p].e, pi/32)
                                and getOrderedEndPoints(tempDict[p0].e, tempDict[p].e)[1] > tempDict[p0].e.length
                                and notCycle):
                                Node.link(tempDict[p0], tempDict[p])
                                self.removeArc(thetaList, t, r)
#                                 print("\t", (int)(t*1000)/1000, r, p0, p, "-", tempDict[p0].e.getEndPointsStr(), "-", tempDict[p].e.getEndPointsStr(), "-")
#                                 print(thetaList)
                                break
                        else:
#                             self.removeArc(thetaMatrix[iR], t, r)
#                             print("\t**", (int)(t*1000)/1000, r, p0, p, "-", tempDict[p0].e.getEndPointsStr(), "-", tempDict[p].e.getEndPointsStr(), "-")
#                             print("**",thetaList)
                            break
#         print()
#         for f in self:
#             print(str(f.pnts[0]) + " " + str(f.pnts[1]) + " " + str(self[f])[-7:] + "\t" + str(f)[-7:])
#             for lnk in self[f].links: 
#                 print("\t", lnk.e.pnts[0], lnk.e.pnts[-1], "\t", str(lnk)[-7:], "\t", str(lnk.e)[-7:] )
    
        for i, f in enumerate(self):
            string = " --- "
            if f in self.ends:
                string = " END "
            
            string = str(i) + string
            
            string += "|" + self[f].e.getEndPointsStr() + "| "
            for l in self[f].links:
                string += " - (" + l.e.getEndPointsStr() + ")"
            string += " -"
            
            print(string)
    
        '''
        This is here because the fiber graph-creation algorithm
        gets stuck in a loop in some images, and I haven't yet figured out why.
        
        '''
        self.drawGraph(links=False)
        temp = self.im.copy()
        self.drawGraph(links=True)
        from main import displayPlots
        try:
            displayPlots([temp, self.im])
        except Exception:
            ()
        exit()
        
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
    
    def __initRadial__(self, imForSize, fibers, maxDist, fiberW):
        '''
        This checks all points within a radius from each point
        Is technically linear, but has a very high constant cost:
            -makes ~60,000 dictionary calls each iteration.
        '''
#         radiusList = np.arange(1.0 , maxDist, 0.5)
#         thetaList0 = np.arange(0 , 2*pi, pi/480)
        radiusList, thetaMatrix0 = self.getradialCircleArray(maxDist)

        for f in fibers:
            self[f] = Node(f)
            
        self.im = np.zeros((len(imForSize), len(imForSize[0])))
        self.fw = fiberW
        self.ends = {}
        self.invSqrRt2 = 1/sqrt(2)
        
        tempDict = {}
        for f in self:
            print(f.pnts[0], f.pnts[-1])
            n = self[f]
            tempDict[f.pnts[0]] = n
            tempDict[f.pnts[-1]] = n
        
        tolerance = pi/8
        
        pointList = list(tempDict)
        
        
        # Sort a list of all points
        
#         i = 0
        for i, p0 in enumerate(pointList):
#             pAvg = (np.array(tempDict[p0].e.pnts[0]) + np.array(tempDict[p0].e.pnts[-1]))/2
#             print(tempDict[p0].e.pnts[0], tempDict[p0].e.pnts[-1], pAvg)

            if i%100 == 0:
                print(i, len(fibers), "building fiber graph")
#             print(i, len(tempDict), "building fiber graph")
#             i+=1
                
# START            
            # this records which points have already been processed, and so don't need to be again.
            flagDict = {}
            
            thetaMatrix = thetaMatrix0.copy()
            
            for iR in range(0, len(thetaMatrix)):
                r = radiusList[iR]
                for iT in range(0, len(thetaMatrix[0])):
                    t = thetaMatrix[iR][iT]
                    if t == -2:
                        break
                    if t == -1:
                        continue
                    p = (p0[0] + int(r * np.array([cos(t), sin(t)])[0]), 
                         p0[1] + int(r * np.array([cos(t), sin(t)])[1]) )
                    if p in flagDict:
                        continue
                    # mark that this point doesn't need to be processed again
                    flagDict[p] = True
                    # if the point is an endpoint to a fiber, but not of the fiber currently being examined
                    
#                     if p in tempDict:
#                         print((int)(t*100)/100, r, p, p0, (p in tempDict), (p != tempDict[p0].pnts[0]), (p != tempDict[p0].pnts[1]))
# END

                    if (p in tempDict) and (p != tempDict[p0].e.pnts[0]) and (p != tempDict[p0].e.pnts[1]):
                        
#                         print("\t", (int)(t*100)/100, r, p, p0)
                        if not (Node.linked(tempDict[p0], tempDict[p])):
#                             print("\t\t", (int)(t*100)/100, r, p, p0)
#                             if notCycle:
                            notCycle = True
                            for n2 in tempDict[p].links:
                                # if the new point is connected to a node already linked to the current node
                                # prevents cycles of order 3
                                if Node.linked(tempDict[p0], n2):
                                    notCycle = False
                                    break
                            
                            if (self.aligned(tempDict[p0].e, tempDict[p].e, pi/32)
                                and getOrderedEndPoints(tempDict[p0].e, tempDict[p].e)[1] > tempDict[p0].e.length
                                and notCycle):
                                Node.link(tempDict[p0], tempDict[p])
                                self.removeArcMatrix(thetaMatrix, t, r)
                                print("\t", (int)(t*1000)/1000, r, p0, p, "-", tempDict[p0].e.getEndPointsStr(), "-", tempDict[p].e.getEndPointsStr(), "-")
#                                 print(thetaList)
                                break
                        else:
#                             self.removeArc(thetaMatrix[iR], t, r)
                            print("\t**", (int)(t*1000)/1000, r, p0, p, "-", tempDict[p0].e.getEndPointsStr(), "-", tempDict[p].e.getEndPointsStr(), "-")
#                             print("**",thetaList)
                            break
#         print()
#         for f in self:
#             print(str(f.pnts[0]) + " " + str(f.pnts[1]) + " " + str(self[f])[-7:] + "\t" + str(f)[-7:])
#             for lnk in self[f].links: 
#                 print("\t", lnk.e.pnts[0], lnk.e.pnts[-1], "\t", str(lnk)[-7:], "\t", str(lnk.e)[-7:] )
    
        for i, f in enumerate(self):
            string = " --- "
            if f in self.ends:
                string = " END "
            
            string = str(i) + string
            
            string += "|" + self[f].e.getEndPointsStr() + "| "
            for l in self[f].links:
                string += " - (" + l.e.getEndPointsStr() + ")"
            string += " -"
            
            print(string)
    
        self.drawGraph(links=False)
        temp = self.im.copy()
        self.drawGraph(links=True)
        from main import displayPlots
        try:
            displayPlots([temp, self.im])
        except Exception:
            ()
        exit()
        
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
    
    #This finds and connects fiber fragments into longer fibers
    def findFibers(self):
        fiberList = []
        tempFiberList = []

        # print all nodes and their links, for debugging
#         for e in self:
#             print("*", e.getEndPoints())
#             temp = -1
#             for n in self[e].links:
#                 print("\t*", n.e.getEndPoints(), n.e in self)
        
        endList = list(self.ends.copy())
        i = 0
        while i < len(endList):
            e = endList[i]
            i+=1
#             print("e: ", e)
            if not e in self.ends or not e in self:
                continue
            
            chain = []
            chain.append(self[e])
            if len(self[e].links) > 0:
                prv = self[e]
                nxt = self[e].links[0]
#                 print(prv.e.getEndPoints())
                p0 = np.array(chain[0].e)
                while True:
#                     print(nxt.e.getEndPoints(), len(endList), [n.e.getEndPoints() for n in nxt.links])
                    temp = -1
    #                 for n in nxt.links:
    #                     print("\t", n.e, sqrDist(nxt.e, n.e))
                    if len(nxt.links) == 2:
                        # this ends the fiber if it has too high a bend, and the points being examined
                        # aren't too close.
    #                     print(p0 - prvP, p0 - nxtP, p0, prvP, nxtP, getAngle(p0 - prvP, nxtP - prvP), sqrDist(chain[0].e, nxt.e), 3*self.fw)
                        ordOrigPrv = getOrderedEndPoints(chain[0].e, prv.e)
                        ordOrignxt = getOrderedEndPoints(chain[0].e, nxt.e)
                        if ((ordOrigPrv[0] > 3*self.fw or chain[0].e == prv.e)
                            and (ordOrignxt[0] > 3*self.fw)):
    #                         and getAngle(prvP - p0, nxtP - p0) < 6*pi/8):
    #                         and getAngle(p0 - prvP, nxtP - prvP) < 6*pi/8):
                            chain.append(nxt)
                            if len(nxt.links) == 1 and nxt.e not in self.ends:
                                self.ends[nxt.e] = nxt
                                endList.append(nxt.e)
#                             print(1)
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
                        nxP = self.getNextPoint(nxt, chain, 5*pi/16)
                        if nxP != 0:
#                             print(nxP.e.getEndPoints())
                            temp = nxP
                    
                    if temp == -1:
                        chain.append(nxt)
                        if len(nxt.links) == 1 and nxt.e not in self.ends:
                            self.ends[nxt.e] = nxt
                            endList.append(nxt.e)
    #                     print(3)
                        break
                    
#                     print(prv.e.getEndPoints(), nxt.e.getEndPoints(), temp.e.getEndPoints())
                    prv = nxt
                    nxt = temp
#                     print("-----")
    #                 print(nxt)
                    chain.append(nxt)
                    
    #                 if nxt.e not in self.ends:
    #                     break
                    
            chain0 = []
            for p in chain:
                chain0.append(p.e)
#             print(chain0)
            Graph.unlinkChain(chain)
            #Could improve speed by making it just prune the parts
            #that changed. That would have to be recursive though. Might be worth it.
            self.prune(endList)
            #don't prune the whole thing, just prune the points that had been touched
#             self.pruneChain()
            
            fPoints = getOrderedEndPoints(chain0[0], chain0[-1])[2:4]
#             print("fPoints: ", fPoints)
            f = Fiber(fPoints, self.fw)
            if f.length > 4:
                tempFiberList.append(f)
#             print("---------------------------------")
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
    
    
#     #This finds and connects fiber fragments into longer fibers
#     def findFibers(self):
#         fiberList = []
#         
# #         print("fibers")
#         for e in self.ends.copy():
# #             print("e: ", e)
#             if not e in self.ends or not e in self:
#                 continue
#             if  len(self[e].links) == 0:
#                 fiberList.append(self[e].e)
#                 continue
#                 
#             chain = []
#             
#             prv = self[e]
#             nxt = self[e].links[0]
#             chain.append(self[e])
#             
#             while True:
#                 temp = -1
# #                 print(nxt.p, len(nxt.links))
#                 if len(nxt.links) == 2:
#                     nxP = self.getNextPoint(nxt, prv, pi/8)
#                     if nxP != 0:
#                         if nxt.links[0] != prv:
#                             temp = nxt.links[0]
#                         else: 
#                             temp = nxt.links[1]
#                     else:
#                         break
#                 elif  len(nxt.links) == 1:
#                     chain.append(nxt)
#                     break
#                 else: #intersection
#                     nxP = self.getNextPoint(nxt, prv, pi/8)
#                     if nxP != 0:
#                         temp = nxP
#                         
#                 if temp == -1:
#                     chain.append(nxt)
#                     break
#                 
#                 prv = nxt
#                 nxt = temp
# #                 print(nxt)
#                 chain.append(nxt)
#                 
#                 if nxt.e not in self.ends:
#                     break
#                 
#             chain0 = []
#             for nF in chain:
#                 chain0.append(nF.e)
# #             print(chain0)
#             Graph.unlinkChain(chain)
#             self.prune()
#             #don't prune the whole thing, just prune the points that had been touched
# #             self.pruneChain()
#             
#             fPoints = getOrderedEndPoints(chain0[0], chain0[-1])[2:4]
#             print(fPoints)
#             f = Fiber(fPoints, self.fw)
#             if f.length > 4:
#                 fiberList.append(f)
#         return fiberList
    
    def prune(self, endList = None):
        pToDelete = []
        pToCut = []
        nToEnds = []
        for p0 in self:
            nodePoint = self[p0]
            if len(nodePoint.links) < 1:
                if self.shouldRemove(nodePoint):
                    pToDelete.append(nodePoint.e)
                
            if (len(nodePoint.links) == 2) and self.inLine(nodePoint.e, nodePoint.links[0].e, nodePoint.links[1].e):
                pToCut.append(nodePoint.e)
    
            if len(nodePoint.links) == 1:
                nToEnds.append(nodePoint)
                
        for p in pToDelete:
            self[p].visited = True
#             if p in self.ends:
#                 self.ends.remove(self[p])
            if p in self.ends:
                del self.ends[p]
            del self[p]
            
        for p in pToCut:
            if (len(self[p].links) == 2):
#                 print("....", p, len(self[p].links))
#                 self[p].printLinks(",,,,")
                Node.cutOut(self[p])
                del self[p]
        
        if endList != None:
            for n in nToEnds:
                # should I do this? Is there a situation where a node will,
                # after other nodes are unlinked around it, still have a single link and yet want to be deleted?
                if n.e not in pToDelete:
                    endList.append(n.e)
                    self.ends[n.e] = n
    
    def getNextPoint(self, node, curChain, angleTol):
        # angleTol is the tolerance on either side. So half of total range.
        minDist = 1000000
        # holds the points which lie nearly directly ahead
        aheadPoints = []
        for x in node.links:
            if x in curChain:
                continue
            minDist, maxDist, p1, p4, p2, p3 = getOrderedEndPoints(node.e, x.e)
            t = getAngle(p1, p4)
            err = t - node.e.angle
            while err > pi:
                err -= 2 * pi 
            while err < -pi:
                err += 2 * pi
            if abs(err) < angleTol:
                aheadPoints.append( (x, abs(err), minDist) )
        if len(aheadPoints) == 0:
            return 0
        aheadPoints = sorted(aheadPoints, key=lambda tup: tup[2])#, reverse=True)
        aheadStr = ""
        for p in aheadPoints:
            aheadStr += str(p[0].e.getEndPoints()) + " " + str(p[1]) + " " + str(p[2]) + " | " 
        print("ahStr", aheadStr, aheadPoints[0])
        return aheadPoints[0][0]

    
    def drawGraph(self, im = 0, links = True):
        from main import drawFibers
        if im == 0:
            im = self.im
        
        # draw all fibers
        drawFibers(im, self, (140, 255))
            
        #draw end fibers
        drawFibers(im, self.ends, (200, 255))
        
        if not links:
            return
        
        # draw links
        visited = {}
        for f1 in self:
            if f1 in visited:
                continue
            for nodeF in self[f1].links:
                visited[nodeF.e] = True
                f2 = nodeF.e
                p1, p2 = getOrderedEndPoints(f1, f2)[-2:]
                rr, cc = line_aa(*p1 + p2)[:2]
                im[rr, cc] = 35  + randint(0, 40)
                im[p1] = 255
                im[p2] = 255
            

    def inLine(self, f0, f1, f2):
        # f1 is the center fiber
        tol = pi/8
        return self.aligned(f0, f1, tol) and self.aligned(f0, f2, tol)
        '''
        Check both ways; if either one fibers angle is similar to the angle made between its 
            points and the avg point of the other, or vise versa. So to make it directionally
            independent, between a large fiber and a skewed segment.
        Or
        Check if the angles between the center's avg and the others ends are all similar
        and then check if ...? Do I need to check anything else?
        I could check, with extra pain and spaghetti, whether those angles match with the average
        so far, but I happen to not like pain, especially with spaghetti on the side.. 
        '''
        
        p01, p02 = f0.getEndPoints()
        p21, p22 = f2.getEndPoints()
        
        t01 = getAngle(f1.pAvg, p01)
        t02 = getAngle(f1.pAvg, p02)
        
        t21 = getAngle(f1.pAvg, p21)
        t22 = getAngle(f1.pAvg, p22)
        
        aligned = ((abs(getAngleDiff(f0.angle, t1)) <= tol) and (abs(getAngleDiff(f0.angle, t2)) <= tol))
        
        return ((abs(getAngleDiff(f0.angle, t1)) <= tol) and (abs(getAngleDiff(f0.angle, t2)) <= tol))
    
        
    
    def aligned(self, f0, f1, tol):
        
        p1, p2 = f1.getEndPoints()
        
        t1 = getAngle(f0.pAvg, p1)
        t2 = getAngle(f0.pAvg, p2)
#         if ((abs(getAngleDiff(f0.angle, t1)) <= tol) and (abs(getAngleDiff(f0.angle, t2)) <= tol)):
#             print(p1, p2, f0.pAvg, getAngleDiff(f0.angle, t1), getAngleDiff(f0.angle, t2), tol)
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
                # I think I don't need this? Its removal improves all output in my test, so
                # its removal should help more often than it hurts
#                 aligned = aligned and (abs(avgAngle - f4.angle) < angleTol)
                
                if endpointsNear and sameSlope and aligned and (newLength > f1.length) and (newLength > f2.length):
                    fiberList.remove(f2)
    #                 f1 = Fiber([far1, far2], fiberW)
                    f1 = f3
                    fiberList[i1] = f1
                    i2 -= 1
                i2 += 1
            i1 += 1
        return fiberList



if __name__ == "__main__":
    from main import cPythonStuff
    cPythonStuff()

