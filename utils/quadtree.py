import numpy as np


class Node():
    def __init__(self, x0, y0, w, h, points, level, n, leaf=False):
        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.h = h
        self.points = points
        self.child = []
        self.level = level
        self.portals = []
        self.isleaf = leaf

    def getw(self):
        return self.w

    def geth(self):
        return self.h

    def getp(self):
        return self.points

    def getportals(self):
        return self.portals


def subdivide(node, k, n):
    if len(node.points) <= k:
        node.isleaf = True
        return

    w_, h_ = node.w / 2, node.h /2
    l = node.level

    p = contains([node.x0, node.y0], w_, h_, node.points)
    x1 = Node(node.x0, node.y0, w_, h_, p, l + 1, n)
    subdivide(x1, k)
    p = contains([node.x0, node.y0 + h_], w_, h_, node.points)
    x2 = Node(node.x0, node.y0 + h_, w_, h_, p, l + 1, n)
    subdivide(x2, k)
    p = contains([node.x0 + w_, node.y0 + h_], w_, h_, node.points)
    x3 = Node(node.x0 + w_, node.y0 + h_, w_, h_, p, l + 1, n)
    subdivide(x3, k)
    p = contains([node.x0 + w_, node.y0], w_, h_, node.points)
    x4 = Node(node.x0 + w_, node.y0, w_, h_, p, l + 1, n)
    subdivide(x4, k)

    node.child = [x1, x2, x3, x4]
    inter_w, inter_h = node.w / n, node.h / n
    node.portals += [[node.x0, node.y0 + i * inter_h] for i in range(n + 1)]
    node.portals += [[node.x0 + inter_w, node.y0] for i in range(n + 1)]
    node.portals += [[node.x0 + node.w, node.y0 + i * inter_h] for i in range(n + 1)]
    node.portals += [[node.x0 + inter_w, node.y0 + i * inter_h] for i in range(n + 1)]


def contains(boundary, w, h, points):
    pts = []
    for i in range(points.shape[0]):
        if points[i][0] >= boundary[0] and points[i][0] <= boundary[0] + w:
            if points[i][1] >= boundary[1] and points[i][1] <= boundary[1] + h:
                pts.append([points[i][0], points[i][1]])
    return np.array(pts)


def find_child(node):
    if not node.child:
        return [node]

    else:
        child = []
        for c in node.child:
            child += (find_child(c))
        return child