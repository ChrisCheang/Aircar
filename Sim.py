import math
import time
import numpy as np
import sympy as smp
from sympy import *
import matplotlib.pyplot as plot

z = symbols("z", real=True)

# Properties of air
R = 287


class Tank:
    def __init__(self, p0, T0, d0, v=0.002):
        self.p0 = p0  # Stagnation properties
        self.T0 = T0
        self.d0 = d0
        self.v = v

    def m(self):
        return (self.p0 * self.v) / (R * self.T0)

    def update(self, time_step, nozzle):
        m_1 = self.m() - nozzle.mdot_max() * time_step
        p_1 = m_1 * R * self.T0 / self.v
        #T_1 = ((self.p0 / p_1) ** 1.4 * self.T0 ** 0.4) ** (1/1.4)        # some big issue with temp calc, assume isothermal for now
        d_1 = p_1 / (R * self.T0)
        return Tank(p_1, self.T0, d_1)


class Nozzle:
    #   design inputs: stagnation properties, back pressure (i.e. pressure to go into turbine)
    #   outputs: maxmimum flow rate, area expansion ratio
    #   too complicated to model the profile on here, so this takes in nozzles that do choke
    #   so isentropic tables and flow relations can be used.

    # change?: fix the throat and exit areas, use that to find mach number at exit using area-mach relation (subsonic
    # case, as the exhaust of the turbine is constricted by the housing + turbine so assume the pressure there will be
    # higher), then calculate the subsonic case pb from that as the output

    # on further thought,

    def __init__(self, p0, T0, d0, pb, throat_diameter=0.0016):      # 0.0016
        self.p0 = p0  # Inlet pressure
        self.T0 = T0  # Inlet temp
        self.d0 = d0  # Inlet density
        self.pb = pb  # back pressure, i.e. pressure going into turbine
        self.throat_diameter = throat_diameter

    def mdot_max(self):
        A_throat = np.pi * (self.throat_diameter / 2) ** 2
        exp = (1.4 + 1) / (2 * (1.4 - 1))
        return self.p0 * A_throat * math.sqrt(1.4 / (R * self.T0)) * (2 / (1.4 + 1)) ** exp

    def q_max(self):
        return self.mdot_max() / self.d0

    def p_critical(self):
        ratio = ((1.4 + 1) / 2) ** (1.4 / (1.4 - 1))  # eq. 8.4 fluids book, 1.8929 for air
        return self.p0 / ratio

    def M_exhaust(self):  # Mach no. of flow out of exit
        ratio = self.p0 / self.pb
        return math.sqrt(5 * (ratio ** (1 / 3.5) - 1))  # inverse of eq. 7.5, one of the isentropic flow relations

    def exhaust_diameter(self):  # required exhaust area to reach/accommodate pb, using equation 8.7
        #exp = (1.4 + 1) / (2 * (1.4 - 1))
        #ratio = ((2 / (1.4 + 1)) * (2 * (1 + (1.4 - 1) * (self.M_exhaust() ** 2) / 2) / (1.4 + 1)) ** exp) / self.M_exhaust()

        ratio = ((5/6) * (1 + 0.2*(self.M_exhaust())**2)) ** 3 / self.M_exhaust()
        A_exhaust = ratio * np.pi * (self.throat_diameter / 2) ** 2
        return math.sqrt(4 * A_exhaust / np.pi)


class TurbineDrive:
    def __init__(self, p0, T0, d0, p1, T1, d1, efficiency=0.1, r=0.03):
        self.p0 = p0  # Inlet pressure
        self.T0 = T0  # Inlet temp
        self.d0 = d0  # Inlet density
        self.p1 = p1  # Outlet pressure
        self.T1 = T1  # Outlet temp
        self.d1 = d1  # Outlet density
        self.efficiency = efficiency
        self.r = r  # radius

    def specific_speed(self, w, Q):
        h = (self.p0 - self.p1) / 98100
        return w * (Q ** (1 / 2)) / ((9.81 * h) ** (3 / 4))

    def power(self, nozzle, efficiency=0.08):
        return efficiency * self.p1 * nozzle.q_max() * math.log(self.p0 / self.p1, np.e)

    def max_w(self, nozzle):
        q = nozzle.q_max()
        blade_swept_area = 0.020 * 0.020  # thickness * radial depth of blades
        tip_speed = q / blade_swept_area
        return tip_speed / self.r


class Aircar:
    def __init__(self, s, v, a, m, t):
        self.s = s  # distance travelled, m
        self.v = v  # velocity, m/s
        self.a = a  # acceleration, m/s^2
        self.m = m  # vehicle mass, kg
        self.t = t  # torque on rear axle, Nm

    def update(self, time_step):
        roll_resist = -0.2  # acceleration due to rolling resistance
        drag = 0.01  # drag proportionality coefficient (not cd, just here for testing reasons)
        wheel_radius = 0.05  # wheel radius in m
        self.a = self.t / (wheel_radius * self.m) + roll_resist - drag * self.v ** 2
        self.v += self.a * time_step
        self.s += self.v * time_step  # could use RK4 for this, but na
        self.t = self.t * 0.995
        if self.t < 0.03:
            self.t = self.t * 0.5
        return Aircar(self.s, self.v, self.a, self.m, self.t)


test_tank = Tank(p0=600000, T0=298, d0=7)
test_nozzle = Nozzle(p0=test_tank.p0, T0=test_tank.T0, d0=test_tank.d0, pb=0.99*test_tank.p0)
test_nozzle.pb = 0.99*test_nozzle.p0
test_turbine = TurbineDrive(p0=test_nozzle.pb, T0=test_nozzle.T0, d0=test_nozzle.pb, p1=100000, T1=298, d1=1)
test_car = Aircar(s=0, v=0, a=0, m=3.0, t=0.1)

clock = 0
time_step = 0.05

end = False
time_list = []
s_l = []
v_l = []

print(test_nozzle.mdot_max())
print(test_nozzle.q_max())
print(test_turbine.power(test_nozzle))
print(test_turbine.max_w(test_nozzle))


def print_stuff_gas():
    print(round(clock, 1), "s,",
          "Tank: ",
          round(test_tank.p0, 0), "Pa, ",
          round(test_tank.T0, 0), "K, ",
          round(1000 * test_tank.m(), 1), "g, ",
          "Nozzle: ",
          round(test_nozzle.mdot_max(), 5), "kg/s, ",
          round(test_nozzle.p0 / test_nozzle.pb, 2), "pressure ratio, ",
          round(test_nozzle.M_exhaust(), 2), "Mach, ",
          round(1000 * test_nozzle.exhaust_diameter(), 5), "mm, ",)


def print_stuff_drive():
    print(round(clock, 1), "s,",
          round(test_car.s, 2), "m,",
          round(test_car.v, 2), "m/s,",
          round(test_car.a, 2), "m/s^2,",
          round(test_car.t, 2), "Nm")


print_stuff_gas()

while not end:
    clock += time_step
    test_car = test_car.update(time_step)
    test_nozzle = Nozzle(p0=test_tank.p0, T0=test_tank.T0, d0=test_tank.d0, pb=0.99*test_tank.p0)
    if test_nozzle.pb < 120000:
        test_nozzle.pb = 100000
    test_tank = test_tank.update(time_step, test_nozzle)
    #print_stuff_drive()
    print_stuff_gas()

    time_list.append(clock)
    s_l.append(test_car.s)
    v_l.append(test_car.v)
    if test_car.s > 25:
        end = True



plot.plot(time_list, s_l)
# plot.show()
plot.plot(time_list, v_l)
# plot.show()
