import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import ctypes
import heapq


class ArrayType:
    def __init__(self, type):
        self.type = type

    def from_param(self, param):
        typename = type(param).__name__
        if hasattr(self, 'from_' + typename):
            return getattr(self, 'from_' + typename)(param)
        elif isinstance(param, ctypes.Array):
            return param
        else:
            raise TypeError('Can\'t convert % s' % typename)

    # Cast from lists / tuples
    def from_list(self, param):
        val = ((self.type) * len(param))(*param)
        return val

    from_tuple = from_list
    from_array = from_list
    from_ndarray = from_list


class Evaluator:
    def __init__(self):
        path = '/geosteiner/libeval.so'
        eval_mod = ctypes.cdll.LoadLibrary(path)
        DoubleArray = ArrayType(ctypes.c_double)
        IntArray = ArrayType(ctypes.c_int)

        eval_mod.gst_open_geosteiner()

        eval_func = eval_mod.eval
        eval_func.argtypes = (DoubleArray, IntArray, ctypes.c_int)
        eval_func.restype = ctypes.c_double
        self.eval_func = eval_func

        gst_rsmt_func = eval_mod.call_gst_rsmt
        gst_rsmt_func.argtypes = (
            ctypes.c_int, DoubleArray, ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
        gst_rsmt_func.restype = ctypes.c_int
        self.gst_rsmt_func = gst_rsmt_func

    def gst_rsmt(self, inputs):
        degree = len(inputs)
        terms = inputs.flatten()
        length = ctypes.c_double()
        nsps = ctypes.c_int()
        sps = (ctypes.c_double * (degree * 2))()
        sps = ctypes.cast(sps, ctypes.POINTER(ctypes.c_double))
        nedges = ctypes.c_int()
        edges = (ctypes.c_int * (degree * 4))()
        ctypes.cast(edges, ctypes.POINTER(ctypes.c_int))
        self.gst_rsmt_func(degree, terms, ctypes.byref(length),
                           ctypes.byref(nsps), sps, ctypes.byref(nedges), edges)
        sp_list = []
        for i in range(nsps.value):
            sp_list.append([sps[2 * i], sps[2 * i + 1]])
        edge_list = []
        for i in range(nedges.value):
            edge_list.append([edges[2 * i], edges[2 * i + 1]])
        return sp_list, edge_list

    def gst_rmst(self, inputs):
        degree = len(inputs)
        terms = inputs.flatten()
        length = ctypes.c_double()
        nsps = ctypes.c_int()
        sps = (ctypes.c_double * (degree * 2))()
        sps = ctypes.cast(sps, ctypes.POINTER(ctypes.c_double))
        nedges = ctypes.c_int()
        edges = (ctypes.c_int * (degree * 4))()
        ctypes.cast(edges, ctypes.POINTER(ctypes.c_int))
        self.gst_rmst_func(degree, terms, ctypes.byref(length),
                           ctypes.byref(nsps), sps, ctypes.byref(nedges), edges)
        edge_list = []
        for i in range(nedges.value):
            edge_list.append([edges[2 * i], edges[2 * i + 1]])
        return edge_list