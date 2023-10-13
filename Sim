import math
import numpy as np
import sympy as smp
from sympy import *
import matplotlib.pyplot as plot


z = symbols("z", real=True)


def h(x):
    if x > 0:
        return x
    else:
        return 0


class Shaft:
    def __init__(self, length, profile, loading_xy, loading_z=False, loading_torsion=False, material="Steel"):
        self.length = length    # float
        self.profile = smp.sympify(profile)    # function (sympy piecewise)
        self.loading_xy = loading_xy    # array of 3d vectors (x, y, z(along shaft)), i.e. n x 3 matrix
        self.loading_z = loading_z  # array of 2d vectors (loading, z(along shaft)), i.e. n x 2 matrix, false boolean if none
        self.loading_torsion = loading_torsion  # array of 2d vectors (torque, z(along shaft)), i.e. n x 2 matrix
        self.material = material

    def shear_force(self, dim, z):
        if dim == "x":
            i = 0
        elif dim == "y":
            i = 1
        sf = 0
        for load in range(len(self.loading_xy)):
            if self.loading_xy[load][2] > z:
                sf -= self.loading_xy[load][i]
        return sf

    def shear_force_combined(self, z):
        sf = math.sqrt(self.shear_force("x", z)**2 + self.shear_force("y", z)**2)
        return sf

    def shear_force_draw(self, dim):
        sf = []
        if dim == "combined":
            for l in range(self.length):
                sf.append(self.shear_force_combined(l))
        else:
            for l in range(self.length):
                sf.append(self.shear_force(dim, l))
        plot.plot(range(self.length), sf)
        plot.show()


test_shaft_loading_xy = np.array([[2, 0, 0], [-2, 1, 80], [-2, -1, 120], [2, 0, 200]])
test_shaft = Shaft(200, 10, test_shaft_loading_xy)     # Piecewise((10, 0 <= z < 100), (20, 100 <= z <= 200))


test_shaft.shear_force_draw("y")
