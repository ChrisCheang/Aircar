import math
import time
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
        self.length = length  # float
        self.profile = smp.sympify(profile)  # function (sympy piecewise)
        self.loading_xy = loading_xy  # array of 3d vectors (x, y, z(along shaft)), n x 3 matrix
        self.loading_z = loading_z  # 2d vector array (loading, z(along shaft)), n x 2 matrix, false boolean if none
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
        sf = math.sqrt(self.shear_force("x", z) ** 2 + self.shear_force("y", z) ** 2)
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
test_shaft = Shaft(200, 10, test_shaft_loading_xy)  # Piecewise((10, 0 <= z < 100), (20, 100 <= z <= 200))

# test_shaft.shear_force_draw("x")
# test_shaft.shear_force_draw("y")


class Aircar:
    def __init__(self, s, v, a, p, m, m_rear, t):
        self.s = s  # distance travelled, m
        self.v = v  # velocity, m/s
        self.a = a  # acceleration, m/s^2
        self.p = p  # tank pressure, Pa
        self.m = m  # vehicle mass, kg
        self.m_rear = m_rear  # traction mass on real axle, kg
        self.t = t  # torque on rear axle, Nm

    def update(self, time_step):
        roll_resist = -0.2  # acceleration due to rolling resistance
        drag = 0.01   # drag proportionality coefficient (not cd, just here for testing reasons)
        wheel_radius = 0.05    # wheel radius in m
        self.a = self.t / (wheel_radius * self.m) + roll_resist - drag * self.v ** 2
        self.v += self.a * time_step
        self.s += self.v * time_step  # could use RK4 for this, but na
        self.t = self.t*0.995
        return Aircar(self.s, self.v, self.a, self.p, self.m, self.m_rear, self.t)


test_car = Aircar(0, 0, 0, 600000, 3.0, 1.5, 0.1)

clock = 0
time_step = 0.05

end = False
time_list = []
s_l = []
v_l = []

while not end:
    clock += time_step
    test_car = test_car.update(time_step)
    print(round(clock, 1), "s,", round(test_car.s, 2), "m,", round(test_car.v, 2), "m/s,", round(test_car.a, 2), "m/s^2,", round(test_car.t, 2), "Nm")
    time_list.append(clock)
    s_l.append(test_car.s)
    v_l.append(test_car.v)
    if test_car.s > 25:
        end = True

plot.plot(time_list,s_l)
plot.show()
plot.plot(time_list,v_l)
plot.show()
