import numpy as np


class GaussPoints:
    def __init__(self, n_gp):
        self.n_gp = n_gp


class GaussPoints_1(GaussPoints):
    def __init__(self):
        GaussPoints.__init__(self, 1)
        self.xw = [[0.], [2.]]


class GaussPoints_2(GaussPoints):
    def __init__(self):
        GaussPoints.__init__(self, 2)
        self.xw = [[-1./np.sqrt(3.), 1./np.sqrt(3.)], [1., 1.]]


class GaussPoints_3(GaussPoints):
    def __init__(self):
        GaussPoints.__init__(self, 3)
        self.xw = [[0., -np.sqrt(3./5.), np.sqrt(3./5.)], [8./9., 5./9., 5./9.]]
