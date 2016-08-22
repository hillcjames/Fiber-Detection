'''
Created on Jul 13, 2015

@author: Christopher Hill

This will count fibers in an image and measure their lengths.

It requires the installation of python3, PIL, image_slicer, cython, matplotlib, scikit-image,
and maybe not iamge_slicer
on linux, you can just do:
    sudo apt-get install python3
    look up how to install pip, AND install it using python3, not python
    use pip to install PIL
    sudo -H python3 -m pip install image_slicer?
    sudo -H pip3 install cython
    sudo -H pip3 install scikit-image
    Or actually just see about installing anaconda 
        https://docs.continuum.io/anaconda/install

Assumes:
    A grayscale image with relatively few large bundles of overlapping fibers 
    
'''

from PIL import Image, ImageDraw
import numpy as np
from scipy import dot, array
from math import cos, sin, pi, sqrt, ceil, atan
import datetime
import os
from json.decoder import NaN
import timeit


from scipy import ndimage, misc
from skimage.feature import peak_local_max
from skimage.draw import line_aa
from matplotlib.pyplot import *
from matplotlib.lines import Line2D

from numpy.core.numeric import ndarray
from subprocess import call
from random import randint

import math
import datetime
import random
from PIL import Image, ImageDraw

from numpy.core.numeric import ndarray
 
from scipy import ndimage, misc
from matplotlib.pyplot import *

from lib.fiberWeb import FiberWeb
from lib.pointWeb import PointWeb
from lib.helperFuncs import *


def horizEdgeArray( w, thickness ):
    try:
        # with fiber width w
        a = np.zeros((2 + thickness, (int)(1.5 * w)))
         
        # x-location of the edge of the fiber
        fiberEdge = (int)(0.25 * w)
         
        # width of the central all-light region
        fiberCenterWidth = (int)(0.75*w)
         
        # x-location of the edge of the central all-light region
        fiberCenterEdge = (int)((len(a[0]) - fiberCenterWidth) / 2)
         
         
        edgePixelValue = -1
        centerPixelVlaue = 4
         
        fiberTotalValue = fiberCenterWidth * centerPixelVlaue + (w - fiberCenterWidth) * edgePixelValue
         
        outsidePixelValue = -1 * (int)(fiberTotalValue / (2 * fiberEdge))
         
#         raise Exception(fiberEdge, fiberCenterEdge, edgePixelValue, centerPixelVlaue)
        for t in range(0, thickness):
            for i in range(0, fiberEdge):
                a[(1 + t, i)] = outsidePixelValue
                a[ (1 + t, len(a[0]) - 1 - i)] = outsidePixelValue
             
            for i in range(fiberEdge, fiberCenterEdge):
                a[(1 + t, i)] = edgePixelValue
                a[ (1 + t, len(a[0]) - 1 - i)] = edgePixelValue
             
            for i in range(fiberCenterEdge, fiberCenterEdge + fiberCenterWidth):
                a[(1 + t, i)] = centerPixelVlaue
                
    except Exception:
        print("Width value too low.")
        raise
    return a

def vertEdgeArray( w, thickness ):
    try:
        # with fiber width w
        a = np.zeros(((int)(1.5 * w), 2 + thickness))
        
        # x-location of the edge of the fiber
        fiberEdge = (int)(0.25 * w)
        
        # width of the central all-light region
        fiberCenterWidth = (int)(0.75*w)
        
        # x-location of the edge of the central all-light region
        fiberCenterEdge = (int)((len(a) - fiberCenterWidth) / 2)
        
        
        edgePixelValue = -1
        centerPixelVlaue = 4
        
        fiberTotalValue = fiberCenterWidth * centerPixelVlaue + (w - fiberCenterWidth) * edgePixelValue
        
        outsidePixelValue = -1 * (int)(fiberTotalValue / (2 * fiberEdge))
    
#         print(fiberEdge, fiberCenterEdge)
        for t in range(0, thickness):
            for i in range(0, fiberEdge):
                a[(i, 1 + t)] = outsidePixelValue
                a[ (len(a) - 1 - i), 1 + t] = outsidePixelValue
            
            for i in range(fiberEdge, fiberCenterEdge):
                a[(i, 1 + t)] = edgePixelValue
                a[ (len(a) - 1 - i), 1 + t] = edgePixelValue
            
            for i in range(fiberCenterEdge, fiberCenterEdge + fiberCenterWidth):
                a[(i, 1 + t)] = centerPixelVlaue
    
    except Exception:
        print("Width value too low.")
        
    return a

def edgeArray( w, degrees, fThickness ):

    try:
        boxW = (int)(1.5 * w)
        if boxW % 2 == 0:
            boxW += 1
        # with fiber width w
        a = np.zeros((boxW, boxW))
#         a = 10*np.ones((boxW, boxW))
        
        # x-location of the edge of the fiber
        fiberEdge = (int)(boxW/6)
        
        # width of the central all-light region
        fiberCenterWidth = (int)(boxW/4) - 1
        
        # x-location of the edge of the central all-light region
        fiberCenterEdge = (int)((boxW - fiberCenterWidth) / 2)
        
        edgePixelValue = 0
        centerPixelVlaue = 40
        
#         fiberTotalValue = fiberCenterWidth * centerPixelVlaue + (w - fiberCenterWidth) * edgePixelValue
#         
#         outsidePixelValue = -1 * (int)(fiberTotalValue / (2 * fiberEdge))
        outsidePixelValue = -30
    
#         print(fiberEdge, fiberCenterEdge)
        for t in range(0, fThickness):
            for i in range(0, fiberEdge):
                a[(i, len(a)/2 - fThickness/2 + 1 + t)] = outsidePixelValue
                a[ (len(a) - 1 - i), len(a)/2 - fThickness/2 + 1 + t] = outsidePixelValue
            
            for i in range(fiberEdge, fiberCenterEdge):
                a[(i, len(a)/2 - fThickness/2 + 1 + t)] = edgePixelValue
                a[ (len(a) - 1 - i), len(a)/2 - fThickness/2 + 1 + t] = edgePixelValue
            
            for i in range(fiberCenterEdge, fiberCenterEdge + fiberCenterWidth + 1):
                a[(i, len(a)/2 - fThickness/2 + 1 + t)] = centerPixelVlaue
    
    except Exception:
        print("Width value too low.")
    
    return ndimage.interpolation.rotate(a, angle = degrees, order = 0)

# def manualConvolve(im, filt):
#     out = np.zeros((len(im[0]), len(im)))
#     for x in range(0, len(im)):
#         for y in range(0, len(im[0])):
#             temp = getFilteredPixVal(im, x, y, filt)
#             out[y][x] = temp
#         print(x)
# #     out /= (ndarray.max(out)/255)
#     return out.as_int()

def circleArray( w ):
    # with fiber width w
    a = np.zeros((w, w))
    a -= 0.5
    
    r = w*0.5
    
    for x in range(0, w):
        for y in range(0, w):
            dSqr = (w*0.5 - x - 0.5)**2 + (w*0.5 - y - 0.5)**2
            print(x, y, dSqr, r**2 )
            if dSqr <= r**2:
                a[x, y] = 0.5
    
    return a

def crossEdgeArray( w ):
    # with fiber width w
    a = np.zeros(((int)(1.5 * w), (int)(1.5 * w)))
    
    # x-location of the edge of the fiber
    fiberEdge = (int)(0.25 * w)
    
    # width of the central all-light region
    fiberCenterWidth = (int)(0.75*w)
    
    # x-location of the edge of the central all-light region
    fiberCenterEdge = (int)((len(a) - fiberCenterWidth) / 2)
    
    
    edgePixelValue = 1
    centerPixelVlaue = 3
    
    fiberTotalValue = fiberCenterWidth * centerPixelVlaue + (w - fiberCenterWidth) * edgePixelValue
    
    outsidePixelValue = -1 * (int)(fiberTotalValue / (2 * fiberEdge))
    
#     print(fiberEdge, fiberCenterEdge)
    for i in range(0, fiberEdge):
        a[i, len(a)//2] = outsidePixelValue
        a[len(a) - 1 - i, len(a)//2] = outsidePixelValue
        
        a[len(a)//2, i] = outsidePixelValue
        a[len(a)//2, len(a) - 1 - i] = outsidePixelValue
    
    for i in range(fiberEdge, fiberCenterEdge):
        a[(i, len(a)//2)] = edgePixelValue
        a[len(a) - 1 - i, len(a)//2] = edgePixelValue
        
        a[len(a)//2, i] = edgePixelValue
        a[len(a)//2, len(a) - 1 - i] = edgePixelValue
    
    
    for i in range(fiberCenterEdge, fiberCenterEdge + fiberCenterWidth):
        a[(i, len(a)//2)] = centerPixelVlaue
        a[len(a)//2, i] = centerPixelVlaue
    
    return a

def fiberBox( size, cen, t, fiberW ):
    # returns a numpy 2d array of width w and height h with 
    #     a fiber-representation going through it at angle t.
    # The fiber has width fiberW.
    # using equation of line ax + by = c
    # and distance formula d = |Ax + By + C| / sqrt(A**2 + B**2), where B = 1

    w, h = size[:]
    box = 64 * np.ones((h,w))
    
    m = math.tan(t)
    b = cen[1] - m * cen[0] #since it passes through the centerpoint
    
    A = -m
    C = -b
    
    denom = 1/math.sqrt(A**2 + 1)
    
    for x in range(0, w):
        for y in range(0, h):
            distToLine = abs(A * x + y + C) * denom
            if distToLine < fiberW / 2:
                box[y][x] = (1 - distToLine/(fiberW/2)) * 172 + 64

            
    return box

def getFilteredPixVal(im, x0, y0, flt):
    # p is the center point, so adjust it
    x0 -= len(flt)//2
    y0 -= len(flt[0])//2
    
    sum = 0
    pixelsVisted = 0

    for x in range(0, len(flt)):
        for y in range(0, len(flt[0])):
            try:
                sum += flt[ x ][ y ] * im[ x0 + x ][ y0 + y ]
                pixelsVisted += 1
            except IndexError:
                ()
#     print(sum)
#     try:
    return sum
#     except ZeroDivisionError:
#         return 0
                
def mergeIms( im1, im2 ):
    if len(im1) != len(im2) or len(im1[0]) != len(im2[0]):
        raise ValueError('Dimension mismatch')
    
    tempIm = ndarray((len(im1), len(im1[0])))

    for x in range(0, len(tempIm[0])):
        for y in range(0, len(tempIm)):
            tempIm[y][x] = im1[y][x] if (im1[y][x] > im2[y][x]) else im2[y][x]
    
    return tempIm
         
def pickyConvolvement( im, f1, f2 ):
#     return im
    
    im1 = ndimage.convolve(im, f1)
    im2 = ndimage.convolve(im, f2)
    
    tempIm = mergeIms(im1, im2)
    
    return tempIm
#     return 255 * tempIm / tempIm.max()

def toBinImg( im, thresh=1 ):
    temp = im.copy()
    ind = im < thresh
    ind0 = im >= thresh
    temp[ind] = 0
    temp[ind0] = 1
    return temp

def toPunctuatedImage( im, sectorSize):
    temp = im.copy()
    
    for c in range(0, 255, sectorSize):
        
        indxs = np.where(np.logical_and(temp >= c, temp < c + sectorSize))
        
        temp[indxs] = c + sectorSize // 2
        
    return temp

def showBigger(im, pixels):
    oW, oH = im.size
    im1 = Image.new('L', (2*oW, 2*oH), 0)
    bigPix = im1.load()
    for x in range(0, oW):
        for y in range(0, oH):
            bigPix[2*x,2*y] = pixels[x,y]
            bigPix[2*x+1,2*y] = pixels[x,y]
            bigPix[2*x,2*y+1] = pixels[x,y]
            bigPix[2*x+1,2*y+1] = pixels[x,y]
    return im1

def printReport(fiberList):
    print(len(fiberList), "fibers were found.")
    
    total = 0
    for i in range(0, len(fiberList)):
        total += fiberList[i].length
    avg = total/len(fiberList)
    
    print("Total length of fibers:", total)
    print("Average length of a fiber:", avg)
    
    sum0 = 0
    for i in range(0, len(fiberList)):
        dif = (fiberList[i].length - avg)
        sum0 += dif*dif
    stdev = sqrt(sum0/len(fiberList))
    
    print("Standard deviation of fiber length:", stdev)

def plotIm( im ):
    imshow(im, cmap='Greys_r', interpolation='none', origin='lower', vmin=0, vmax=255,
            extent=(0, len(im[0]), 0, len(im))
           )

def displayPlots( ims):
    fig = figure(1)
    
    if len(ims) == 0:
        show()
        return
    
    w = int( sqrt(2*len(ims)) )
    h = int(len(ims) / w)
    while w*h < len(ims):
        if min([w, h]) == w:
            w += 1
        else:
            h += 1
        
    i = 0
    for im in ims:
        ax = fig.add_subplot(h, w, i + 1)
        axis('off')
        plotIm(im)
#         extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#         savefig("displayedIm" + str(int(im[len(im)/2][len(im[0])/2]))
#                               + str(int(im[len(im)/2 + 1][len(im[0])/2 + 1])),
#                                bbox_inches=extent, pad_inches=0)
        i += 1
        
    show()
    i = 0
    fig2 = figure()
    for im in ims:
#         fig2 = figure()
#         ax2 = fig2.add_subplot(1, 1, 1, aspect='normal')
#         axis('off')
# #         plotIm(im)
#         extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# #         print(str(im)[26:30])
#         savefig("displayedIm" + str(im)[23] + str(im)[29], bbox_inches=extent)
        axis('off')
        
        fig2.set_size_inches(len(im[0]), len(im))
         
        ax = Axes(fig2, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig2.add_axes(ax)
        imshow(im, cmap='Greys_r', interpolation='none', origin='lower', vmin=0, vmax=255,
                extent=(0, len(im[0]), 0, len(im)), aspect='normal'
               )
        fName = (os.path.join("autoSavedIms", "displayedIm") + str(int(np.sum(im))%1000) + "_" + str(i) + ".png")
        #int(im[len(im)/2][len(im[0])/2]))
        #    + str(int(im[len(im)/2 + 1][len(im[0])/2 + 1])
        savefig(fName, dpi = 1)
        i += 1
    
def findPath(im, p1, p2):
    return True
#     print(ndarray.mean(im), ndarray.std(im))
#     displayPlots([(im > ndarray.mean(im) + ndarray.std(im)/2)*128])
#     print(1/0)
    points = getConnectingPoints(p1, p2)
    
    goodPoints = 0
    
    for p in points:
        if im[p] > 0:
           goodPoints += 1
#     print("Goodpoints btw", p1, "and", p2, ":", goodPoints/len(points))
    
    return goodPoints/len(points) > 0.8

def getCol(depth):
    if depth % 3 == 0:
        return [1, 0, 0]
    elif depth % 3 == 1:
        return [0, 1, 0]
    else:
        return [0, 0, 1]

def splitDraw(im, out, x1, x2, y1, y2, depth):
#     print(((2 - depth)*"\t"), "Split,", x1, x2, y1, y2, depth)
    if (depth == 0) or (y1 == y2) or (x1 == x2):
        return
    depth -= 1
    region = im[y1:y2, x1:x2]
#     print(region)
#     print(len(im), len(im[0]), y2-y1, x2-x1, ndarray.mean(region))
    uniformity = np.abs(ndarray.mean(region)*2 - 1)
    print("U:", uniformity)
    if uniformity > 0.97:
        return
    yC, xC = np.array(ndimage.measurements.center_of_mass(region)).astype(int)
#     print(yC, xC)
    for x in range(x1, x2):
        out[y1 + yC][x] = getCol(depth)
    for y in range(y1, y2):
        out[y][x1 + xC] = getCol(depth)
#     print((1 - depth)*"\t_", x1, xC, x2, y1, yC, y2)
    splitDraw(im, out, x1, x1 + xC, y1, y1 + yC, depth )
    splitDraw(im, out, x1 + xC, x2, y1, y1 + yC, depth )
    splitDraw(im, out, x1, x1 + xC, y1 + yC, y2, depth )
    splitDraw(im, out, x1 + xC, x2, y1 + yC, y2, depth )
    return  

def split(im, out, x1, x2, y1, y2, depth):
#     print(((2 - depth)*"\t"), "Split,", x1, x2, y1, y2, depth)
    if depth == 0:
#         out[(y1 + y2)/2, (x1 + x2)/2] = getCol(depth)
        return
    if (y1 == y2) or (x1 == x2):
        return
    depth -= 1
    region = im[y1:y2, x1:x2]
#     print(len(im), len(im[0]), y2-y1, x2-x1, ndarray.mean(region))
    uniformity = np.abs(ndarray.mean(region)*2 - 1)
    print("U:", uniformity)
    if uniformity > 0.95:
        return
    yC, xC = np.array(ndimage.measurements.center_of_mass(region)).astype(int)
    print((1 - depth)*"\t_", x1, xC, x2, y1, yC, y2)
    out[y1 + yC, x1 + xC] = getCol(depth)
    split(im, out, x1, x1 + xC, y1, y1 + yC, depth )
    split(im, out, x1 + xC, x2, y1, y1 + yC, depth )
    split(im, out, x1, x1 + xC, y1 + yC, y2, depth )
    split(im, out, x1 + xC, x2, y1 + yC, y2, depth )
    return  

def getLineCoeffs(p1, p2):
    print(p1, p2, p1[1]*p2[0], p2[1]*p1[0])
    a = (p2[1] - p1[1])/(p1[0]*p2[1] - p2[0]*p1[1] + 0.01)
    b = (p2[0] - p1[0])/(p1[1]*p2[0] - p2[1]*p1[0] + 0.01)
    return a, b
#     print(a, b)
#     if abs(a) < abs(b):
#         for x in range(p1[0], p2[0] + 1):
#             y = 1/b - (a/b)*x
#     else:
#         for y in range(p1[1], p2[1] + 1):
#             x = 1/a - (b/a)*y
#     return (y, x)


# data = np.random.random(100)
# bins = np.linspace(0, 1, 10)
# digitized = np.digitize(data, bins)
# bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]
# print(data)
# print(bins)
# print(digitized)
# print(bin_means)
# print(1/0)

def binFibers(fiberSegments, imW, imH):
    '''
    This will store bins in a dictionary, allowing for range queries by dividing out the
    lower numbers of the keys of access.
    The bin will be found using the bin dictionary, and then that bin's count will be incremented.
    At the end, the counts will be used to find local maximums of line likelihood.
    The line will be traced from left to right until it gets close to a segment, and will continue
    looking ahead until it goes off-screen or stops being some distance close to a segment.
    Segments in that bin, that is.
    Remember to only look ahead for a lack of a segment, don't adjust the new endpoint until you've
    found a segment to extend the line to.
    '''
    
    numBins = 100

    
    binA = np.array([f.a for f in fiberSegments])
#     for f in fiberSegments:
#         print(f.a)
    
    
    bins = np.linspace(0, 1, numBins)
    digitized = np.digitize(binA, bins)
    bin_means = [binA[digitized == i] for i in range(0, len(binA))]
#     for i in range(0, len(binA)):
#         print("::", binA[digitized == i], digitized[i] )
#     print(digitized)
#     bins = {}
#     for f in fiberSegments:
#         A = int(f.a*numBins)
#         B = int(f.b*numBins)
        
#         bin = Bin(a, b)
#         if not (A, B) in bins:
#             bins[(A, B)] = bin
#         bins[(A, B)].count =+ 1
#         print(a, b, A, B, bins[(A, B)].count)
#     for bin in binA:
#         print(binA[bin].a, binA[bin].a, binA[bin].count)

#     for x in binA:
#         print("_" , x)
#     for y in bin_means:
#         print(y)

def drawFibers(im, fibers):
    for f in fibers:
#         print(f, f.pnts[0], f.pnts[len(f.pnts) - 1])
        p1 = f.pnts[0]
        p2 = f.pnts[-1]
        
        rr, cc, val = line_aa(*p1 + p2)
        im[rr, cc] = 55  + randint(0, 1)
        
    for f in fibers:
        p1 = f.pnts[0]
        p2 = f.pnts[-1]
        im[p1] = 255
        im[p2] = 255

def tempGraphStuff(im0, distanceIm, l, fw):
#     fw = 2
#     l = [(4, 4), (5, 5), (13, 5), (5, 15), (15, 15), (10, 10), (20, 20), (5, 28)]
#     l = [(4, 4), (6, 4), (6, 5), (3, 2), (8, 5), (9, 7)]
#     im = np.zeros((10, 10))

    print("About to create graph")
    pWeb = PointWeb(distanceIm, l, 1.4*sqrt(5), fw) # 4*sqrt(5) is about 8.9
    
    print("...Done creating graph")
    
#     im2 = pWeb.im.copy()
#     
#     for p1 in pWeb.connectedEnds:
#         for p2 in pWeb.connectedEnds[p1].links:
#             print("con", p1, p2.p)
#             rr, cc = line_aa(*p1 + p2.p)[:2]
#             im2[rr, cc] = 55  + randint(0, 1)
    
    
    im1 = pWeb.im.copy()
    im2 = pWeb.im.copy()
    im3 = pWeb.im.copy()
    
    pWeb.drawGraph()
    
    
    fibers1 = pWeb.findFibers()
#     displayPlots([pWeb.im])
#     exit()
    
    FiberWeb.fixBrokenFibers(fibers1, 3*fw, pi/8)
    
#     fWeb = FiberWeb(distanceIm, fibers1, 5*fw, fw)
      
#     fibers2 = fWeb.findFibers()
#     drawFibers(im3, list(fWeb))
#     for f in fWeb:
#         # draw links
#         for n in fWeb[f].links:
#             p1, p2 = getOrderedEndPoints(f, n.e)[4:6]
#             c = randint(40,80)
#             for p in getConnectingPoints(p1, p2):
#                 im3[p] = c

    drawFibers(im1, fibers1)
#     drawFibers(im2, fibers2)

    
    print("Displaying plots")
    displayPlots([pWeb.im, im1])
#     displayPlots([pWeb.im, im1, im2, im3])
#     displayPlots([im0, pWeb.im, distanceIm*128, im1, im2])
    print("Done displaying plots")
    
    
    return

# when traversing, give nodes a value that is unique for every fiber trace.
# to act as a 'visited' flag, without needing to reset all flags between fiber traces.


def filterImage( im0, fw):
#     im0 = ndimage.gaussian_filter(im.copy(), sigma=2)
#     blankIm = np.zeros((len(im0), len(im0[0])))

    fH = 5 * horizEdgeArray(fw, 1)
    fV = 5 * vertEdgeArray(fw, 1)
     
    res0 = pickyConvolvement(im0.astype(int), fH, fV)
    res0[res0 < 0] = 0
    res0[res0 > 255] = 255
    res1 = 256*toBinImg(res0, 120)
     
    im = toBinImg(res1, 200)
    
    distance = ndimage.distance_transform_edt(im)
    
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=im)
    
    #remove all L-shapes
    f0 = np.zeros((3,3))
    f0[0][0] = 1
    f0[2][0] = 1
    f0[1][2] = 1
    f1 = np.rot90(f0)
    f2 = np.rot90(f1)
    f3 = np.rot90(f2)
    f4 = ndimage.generate_binary_structure(2, 1).astype(int)
    f4[1,1] = 0
    f4[1] *= 2

    temp0 = toBinImg(local_maxi - ndimage.convolve(local_maxi, f0)//2)
    temp1 = toBinImg(temp0 - ndimage.convolve(temp0, f1)//2)
    temp2 = toBinImg(temp1 - ndimage.convolve(temp1, f2)//2)
    temp3 = toBinImg(temp2 - ndimage.convolve(temp2, f3)//2)
    final = toBinImg(temp3 - ((ndimage.convolve(temp3, f4)%4) + 1 )//4)
#     ims = [ im0, local_maxi * 64, temp3*64, final*64 ]
#     displayPlots([(im > ndarray.mean(im) + ndarray.std(im))*64 + final * 128])
#     return
    maxPoints = np.nonzero(final)
     
    maxPoints = list(zip(maxPoints[0], maxPoints[1]))
    
    tempGraphStuff(im0, final, maxPoints, fw)
    return final

def drawLine(im, p1, p2):
    a = (p2[1] - p1[1])/(p1[0]*p2[1] - p2[0]*p1[1] - 0.1)
    b = (p2[0] - p1[0])/(p1[1]*p2[0] - p2[1]*p1[0] - 0.1)
    print(a, b)
    if abs(a) < abs(b):
        for x in range(p1[0], p2[0] + 1):
            y = 1/b - (a/b)*x
            print(y,x)
            im[(int(y), int(x))] = 255
    else:
        for y in range(p1[1], p2[1] + 1):
            x = 1/a - (b/a)*y
            print(y,x)
            im[(int(y), int(x))] = 255

def cPythonStuff():
#     call(["python3", "setup.py", "build_ext", "--inplace"])
#     im0 = fiberBox((20, 20), (10, 10), 3*pi/4, 5)
#     im0 = fiberBox((40, 20), (10, 15), 4*pi/7, 5)
#     im0 = mergeIms( fiberBox((400, 200), (100, 150), 4*pi/7, 10), fiberBox((400, 200), (300, 150), 3*pi/9, 10) )


    im0 = mergeIms( fiberBox((40*5, 20*5), (10*5, 15*5), 4*pi/7, 5), fiberBox((40*5, 20*5), (30*5, 15*5), 8*pi/7, 5) )
    im1 = mergeIms( fiberBox((40*5, 20*5), (30*5, 15*5), 2*pi/7, 5), fiberBox((40*5, 20*5), (20*5, 15*5), 7*pi/9, 5) )
    im2 = mergeIms( fiberBox((40*5, 20*5), (0, 8*5), 1.2*pi/7, 5), fiberBox((40*5, 20*5), (35*5, 0), 2*pi/9, 5) )
    im3 = mergeIms( fiberBox((40*5, 20*5), (0, 8*5), 3*pi/7, 5), fiberBox((40*5, 20*5), (2*5, 100), -1*pi/9, 5) )
#     displayPlots([im2])
#     exit()
    im5 = mergeIms(im0, im1)
    im6 = mergeIms(im2, im3)
    im0 = mergeIms(im5, im6)
#     displayPlots([im0])
#     exit()
    filterImage(im0, 5)
    return

    im0 = np.array( Image.open("Images/smallTest2.jpg") )
    im0 = ndimage.gaussian_filter(im0.copy(), sigma=3)
    filterImage(im0, 10)
    print("here")
    return

#     x, y = np.indices((5, 5))
#     x1, y1, = 2, 2
#     r1 = 1.2
#     mask_circle = (x - x1) ** 2 + (y - y1) ** 2 < r1 ** 2
#     print(mask_circle)
#     local_maxi = peak_local_max(distance, indices=False, footprint=mask_circle, labels=im)
    
#     im1 = toBinImg(distance.copy(), 2)

#     findFibersWeb(im)
    
    
#     l = [im0, 255*im, 255*(distance/ndarray.max(distance)), 255*(im1/ndarray.max(im1))]
#     displayPlots(l[1::2])
#     displayPlots([im0])
#     pyplot.imshow(255*out1)
#     show()
    
#     print(out)
    return
    
    
#     im1 = mergeIms(fiberBox(dimensions, (40, 80), 7*pi/8, 10), fiberBox(dimensions, (180, 130), pi/4, 10))
#     im2 = mergeIms(fiberBox(dimensions, (150, 70), 0*pi/8, 10), fiberBox(dimensions, (20, 180), 5*pi/8, 10))
#     im3 = mergeIms(im1, fiberBox(dimensions, (150, 70), 4*pi/8, 10))
#     im = mergeIms(im2, im3)
#     temp = np.zeros(5)
#     temp[1] = 1
#     temp[2] = 2
#     temp[3] = 1
#     indx = temp >= 1 
#     print(temp, ndarray.sum(indx))
#     raise Exception("\nsome way to determine if, when a threshhold is applied to a given square \
# as it is examined,\n the resulting boolean matrix is close enough to what it \
# should be.")
#     
#     return

    im = np.array( Image.open("Images/smallTest2.jpg") )
    im0 = im.copy()

    im = ndimage.gaussian_filter(im.copy(), sigma=2)
    
    
    fw = 10
    fH = 5 * horizEdgeArray(fw, 1)
    fV = 5 * vertEdgeArray(fw, 1)
    
    res0 = pickyConvolvement(im.astype(int), fH, fV)
    res0[res0 < 0] = 0
    res0[res0 > 255] = 255
    res0 = 256*toBinImg(res0, 120)
    res1 = ndimage.median_filter(res0, size=8)
#     findFibersWeb()
    displayPlots([res0, res1])
    return
    
      
#     sx = ndimage.sobel(im, axis=0, mode='constant')
#     sy = ndimage.sobel(im, axis=1, mode='constant')
#     sob = np.hypot(sx, sy)
#       
#     bin0 = toBinImg(im0.copy(), 31)
#     bin1 = toBinImg(res2.copy(), 131)
#       
#     close_bin0 = ndimage.binary_closing(bin0)
#     close_bin1 = ndimage.binary_closing(bin1)
#       
#     close_open_bin0 = ndimage.binary_opening(close_bin0)
#     close_open_bin1 = ndimage.binary_opening(close_bin1)
#     subIms = [bin0, bin1]
#     displayPlots( subIms )


'''
First off, get clean binary image.
Then create a network of connected white spots.
    Overlay a larger grid. At each intersection, check for whiteness (in binary img)
    if white, check immediate surrounding area (on grid) for whiteness.
    Repeat, unless point has been processed already.
    If found, create add link to the second spot, to the first one.
    So points now hold links to all the white spots near them.
    Remove all points with exactly two links in a relative line through the center.
        like A-B-C-D --> A-C-D --> A-D
    So now have a list of endpoints and intersections.
    Go through endpoints.
        Follow links, in the straightest line you can, until you either hit another endpoint
        or there are no more links close enough to being in front of you.
        Save that thing as a line.
        Remove all links used by the fiber.
        If what was an intersection now has only two links, remove it like before.
        If it only has one, add it to the endpoints list.
        If it has zero, remove it entirely.

'''


if __name__ == "__main__":
#     import cProfile
#     cProfile.run('cPythonStuff()')
    cPythonStuff()
    
#     d1 = datetime.datetime.now()

#     im = BigImage("","rainbow.jpg",9)
#     im = BigImage('Images/smallTest.jpg')
#     main(10, "Images/","colorAdjustedSmallTest.tif")
#     test()
#     main(10, "Images/","smallTest.jpg")

#     main(8, "Images/","smallTest2.jpg")

#     main(10, "Images", "midSizedTest.jpg")
#     main(10, 'Images/CarbonFiber/', 'GM_LCF_EGP_23wt%_Middle_FLD1(circleLess).tif')

#     d2 = datetime.datetime.now()
#     print('Running time:', d2-d1)



    