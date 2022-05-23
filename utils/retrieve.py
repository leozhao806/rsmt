from geo_utils import *
import numpy as np


class picker():
    def __init__(self, points, portals, likelihoods):
        self.points = points
        self.portals = portals
        self.eval = Evaluator()
        self.likelihoods = likelihoods

    def retrieve(self, th, delta):
        opt = float(1e9)
        idx = np.where(self.likelihoods >= th)
        picked_portals = self.portals[idx]
        st = self.eval.gst_rsmt(np.vstack(self.points, picked_portals))
        prev = opt
        prev_picked = self.portals
        while st < prev or th > 0:
            prev = st
            prev_picked = picked_portals
            th -= delta
            idx = np.where(self.likelihoods >= th)
            picked_portals = self.portals[idx]
            st = self.eval.gst_rsmt(np.vstack(self.points, picked_portals))

        if th == 0:
            return picked_portals
        else:
            return prev_picked