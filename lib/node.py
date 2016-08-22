'''
Created on May 14, 2016

@author: james
'''

class Node(object):
    def __init__(self, e0):
            self.e = e0
            self.links = []
            self.end = False
            self.visited = False
             
    def getStatus(self):
        return len(self.links)
    
    def printLinks(self, prefix):
        for point in self.links:
            print(prefix, point.p)
         
    @staticmethod
    def linked(n1, n2):
        return n2 in n1.links
         
    @staticmethod
    def link(n1, n2):
        if not Node.linked(n1, n2):
            n1.links.append(n2)
        if not Node.linked(n2, n1):
            n2.links.append(n1)
         
    @staticmethod
    def unlink(n1, n2):
        n1.links.remove(n2)
        n2.links.remove(n1)
         
    @staticmethod
    def cutOut(n):
        if len(n.links) != 2:
            # happens when an isolated cycle is reduced down from 3 points to 2
            # because removing one doesn't link the remaining two together again,
            # because they're already linked. SO they're now linked just to each
            # other - and therefore only have one link each.
            # should rarely happen, since linked points must be close to co-linear 
            return
        
        # to be called on points with only two links, in a line
        n1 = n.links[0]
        n2 = n.links[1]
        Node.unlink(n1, n)
        Node.unlink(n, n2)
        Node.link(n1, n2)
        
