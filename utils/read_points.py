import numpy as np
from quadtree import *
from geo_utils import *
import torch
import dgl

def read_points(file):
    with open(file, 'r') as f:
        points = []
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(',')
            line = list(map(int, line))
            points.append(line)
    return points

def process_points(point):
    node = Node(0, 0, 500, 500, point, 0, 50)
    subdivide(node, 30, 50)
    graphs = []
    lgraphs = []
    queue = [node]
    cp = point
    while queue:
        _node = queue.pop()
        portals = _node.portals()
        g = portal_graphs(portals)
        graphs.append(g)
        if node.isleaf:
            bgraph = base_graphs(_node.points)
            lgraphs.append(bgraph)
        for c in _node.child:
            queue.append(c)
    return graphs, lgraphs, cp

def portal_graphs(portals):
    dist = abs(portals[0][1] - portals[1][1])
    g = dgl.graph()
    g.add_nodes(list(range(portals)))
    g.ndata = torch.tensor(portals)
    src = list(range(g.number_of_nodes()))
    dst = [i + 1 for i in src]
    dst[-1] = 0
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    g.edata = torch.ones(g.number_of_edges()) * dist
    return g

def base_graphs(points):
    eval = Evaluator()
    sp, edges = eval.gst_rsmt(points)
    p = points + sp
    g = dgl.graph()
    g.add_nodes(list(range(len(p))))
    g.ndata = torch.tensor(p)
    src = [edge[0] for edge in edges]
    dst = [edge[1] for edge in edges]
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    return g