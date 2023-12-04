import math
import numpy as np
from sympy import *
import matplotlib.pyplot as plot

z = symbols("z", real=True)

# Properties of air
R = 287
p_ambient = 100000

# Some vehicle properties
step_down = 4.58
throat_diameter = 0.002  #0.0016
turbine_radius = 0.04

class Tank:
    def __init__(self, p0, T0, d0, v=0.002):
        self.p0 = p0  # Stagnation properties
        self.T0 = T0
        self.d0 = d0
        self.v = v

    def m(self):
        return (self.p0 * self.v) / (R * self.T0)

    def stored_energy_isothermal(self):
        # isothermal case for overall consideration reasonable (?) as discharge is relatively long at 8 sec AND is conservative (i.e. less energy)
        return self.p0 * self.v * math.log(self.p0 / p_ambient, np.e)

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

    # on further thought, no - design around the output exit dia.

    def __init__(self, p0, T0, d0, pb, throat_diameter=throat_diameter, discharged=False):      # 0.0016
        self.p0 = p0  # Inlet pressure
        self.T0 = T0  # Inlet temp
        self.d0 = d0  # Inlet density
        self.pb = pb  # back pressure, i.e. pressure going into turbine
        self.throat_diameter = throat_diameter
        self.discharged = discharged  # A boolean to model if the discharging is complete

    def mdot_max(self):
        A_throat = np.pi * (self.throat_diameter / 2) ** 2
        exp = (1.4 + 1) / (2 * (1.4 - 1))
        return self.p0 * A_throat * math.sqrt(1.4 / (R * self.T0)) * (2 / (1.4 + 1)) ** exp

    def q_max(self):             # this is constant!
        return self.mdot_max() / self.d0

    def p_critical(self):
        ratio = ((1.4 + 1) / 2) ** (1.4 / (1.4 - 1))  # eq. 8.4 fluids book, 1.8929 for air
        return self.p0 / ratio

    def M_exhaust(self):  # Mach no. of flow out of exit
        ratio = self.p0 / self.pb
        if not self.discharged:
            return math.sqrt(5 * (ratio ** (1 / 3.5) - 1))  # inverse of eq. 7.5, one of the isentropic flow relations
        else:
            return 0

    def exhaust_diameter(self):  # required exhaust area to reach/accommodate pb, using equation 8.7
        ratio = ((5/6) * (1 + 0.2*(self.M_exhaust())**2)) ** 3 / self.M_exhaust()
        A_exhaust = ratio * np.pi * (self.throat_diameter / 2) ** 2
        if not self.discharged:
            return math.sqrt(4 * A_exhaust / np.pi)
        else:
            return "discharged"


class TurbineDrive:
    # currently this uses a somewhat more empirical approach (as I have no idea how to do it otherwise)
    # , using data, findings and assumptions present in this video: https: // www.youtube.com / watch?v = RMUHxo2TOUk
    # The efficiency is first estimated to be 10% (roughly similar to that found in the above video), and maximum
    # turbine power is calculated as efficiency x integral of pdv of an isothermal process covering the inlet and
    # outlet conditions (lower pressure drop so reasonable assumption, can be updated later). The no-load rpm is then
    # estimated using the speed of a column of gas with nozzle q_max and ambient density (air expands and comes out of
    # blades at atm) and the "swept" area of the blades as the tip speed. A power curve similar to that in the video
    # is then extrapolated using these two values, which can then be used to calculate torque curves etc.

    def __init__(self, p0, T0, d0, p1, T1, d1, nozzle, efficiency=0.1, r=turbine_radius):
        self.p0 = p0  # Inlet pressure
        self.T0 = T0  # Inlet temp
        self.d0 = d0  # Inlet density
        self.p1 = p1  # Outlet pressure
        self.T1 = T1  # Outlet temp
        self.d1 = d1  # Outlet density
        self.nozzle = nozzle
        self.efficiency = efficiency
        self.r = r  # radius

    def specific_speed(self, w):
        h = ((self.p0 - self.p1) / 100000) / 0.0981
        Q = self.nozzle.q_max()
        return w * (Q ** (1 / 2)) / ((9.81 * h) ** (3 / 4))

    def max_power(self):
        if self.p1 < self.p0:
            return self.efficiency * self.p1 * self.nozzle.q_max() * math.log(self.p0 / self.p1, np.e)   # method from video, isothermal
            #return self.efficiency * (self.p0 - self.p1) * self.nozzle.q_max()    # method from P.J.
        else:
            return 0

    def no_load_w(self):
        blade_swept_area = 0.012 * 0.010  # thickness * radial depth of blades
        blade_thickness_ratio = 0.0    # how much of the space is taken up by the blades
        tip_speed = ((self.nozzle.mdot_max() / (1 - blade_thickness_ratio)) / self.d1) / blade_swept_area
        return tip_speed / self.r

    def no_load_rpm(self):
        return self.no_load_w() * 60 / (2 * np.pi)

    def torque(self, rpm):
        # assumption: maximum power occurs at 50% of no_load_rpm
        # assumption: torque curve is linear

        half_no_load_rpm_torque = self.max_power() / (0.5 * self.no_load_w())
        stall_torque = 2 * half_no_load_rpm_torque
        if rpm < self.no_load_rpm():
            return stall_torque - (stall_torque * rpm / self.no_load_rpm())
        else:
            return 0


class Aircar:
    def __init__(self, s, v, a, m, rpm, t, drive, step_down, r=0.0401, kinetic_energy=0):
        self.s = s  # distance travelled, m
        self.v = v  # velocity, m/s
        self.a = a  # acceleration, m/s^2
        self.m = m  # vehicle mass, kg
        self.rpm = rpm   # wheel rpm
        self.t = t  # torque on rear axle, Nm
        self.drive = drive
        self.step_down = step_down    #step down ratio in gearbox
        self.r = r  # rear wheel radius
        self.rrc = 0.015  # rolling resistance coefficient, https://www.matec-conferences.org/articles/matecconf/pdf/2019/03/matecconf_mms18_01005.pdf
        self.drivetrain_efficiency = 0.85 #  efficiency with account of drivetrain loss
        self._kinetic_energy = kinetic_energy

    @property
    def set_kinetic_energy(self):
        linear = 0.5 * self.m * self.v ** 2
        # for rotational inertia, consider rotational inertia of turbine, front wheels and gears (shafts have small R so consider negligible, rear wheels are small and light)
        I_turbine = 0.5 * 0.044 * 0.04 ** 2  # mass from CAD
        I_wheel = I_turbine  # similar size and weight to save time
        I_Idler = 0.5 * 0.28 * 0.025 ** 2  # mass from CAD
        I_drive = 0.5 * 0.34 * 0.02752

        KE_rotational_turbine = 0.5 * I_turbine * (self.step_down * self.rpm * 2 * np.pi / 60) ** 2
        KE_rotational_wheels = 2 * 0.5 * I_wheel * (self.rpm * 2 * np.pi / 60) ** 2
        KE_rotational_idler = 0.5 * I_Idler * (self.rpm * 2 * np.pi / 60) ** 2
        KE_rotational_drive = 0.5 * I_drive * (self.rpm * 2 * np.pi / 60) ** 2

        KE_rotational = KE_rotational_turbine + KE_rotational_wheels + KE_rotational_idler + KE_rotational_drive
        KE_translational = 0.5 * self.m * self.v ** 2
        self._kinetic_energy = KE_translational + KE_rotational

    def get_kinetic_energy(self):
        return self._kinetic_energy

    def del_kinetic_energy(self):
        del self._kinetic_energy

    kinetic_energy = property(get_kinetic_energy, set_kinetic_energy, del_kinetic_energy, "kinetic energy")

    def inertia_total(self):
        I_turbine = 0.5 * 0.044 * 0.04 ** 2  # mass from CAD
        I_wheel = I_turbine  # similar size and weight to save time
        I_idler = 0.5 * 0.28 * 0.025 ** 2  # mass from CAD
        I_drive = 0.5 * 0.34 * 0.02752
        I_total = I_turbine * self.step_down + 2 * I_wheel + I_idler + I_drive
        return I_total

    def frictional_torque(self):
        rolling_resistance_torque = self.rrc * self.m * self.r * 9.81
        radial_load = self.m / 4  # assume 50-50 and symmetric weight distribution
        M_frictional = 0.00097  # number obtained from SKF bearing tool, from which starting torque is negligible
        return M_frictional * 4 + rolling_resistance_torque

    def update(self, time_step, drive, ratio):      # change in mass of air canister negligible
        self.s += self.v * time_step  # could use RK4 for this, but na
        self.v += self.a * time_step
        if self.v >= 0:  #impulse momentum method; check one of the pages on the back of my orange A5 notebook
            self.a = self.t * self.r / (self.inertia_total() + self.m * self.r**2)
        else:
            self.v = 0
        self.rpm = (self.v / self.r) * 60 / (2 * np.pi)
        self.t = drive.torque(self.rpm * self.step_down) * self.step_down * self.drivetrain_efficiency - self.frictional_torque()
        return Aircar(s=self.s, v=self.v, a=self.a, m=self.m, rpm=self.rpm, t=self.t, drive=self.drive, step_down=ratio)

#'''

clock = 0
time_step = 0.05

tank = Tank(p0=650000, T0=298, d0=7)
nozzle = Nozzle(p0=tank.p0, T0=tank.T0, d0=tank.d0, pb=0.995 * tank.p0)
turbine = TurbineDrive(p0=nozzle.pb, T0=nozzle.T0, d0=nozzle.d0, p1=p_ambient, T1=298, d1=1, nozzle=nozzle)    #the d0 is wrong but not used so ignore for now
#carpre = Aircar(s=0, v=0, a=0, m=3.0, rpm=0, t=turbine.torque(0)*step_down, drive=turbine) #  weird method but use this to calculate starting net torque
car = Aircar(s=0, v=0, a=0, m=3.0, rpm=0, t=turbine.torque(0) * step_down * 0.85, drive=turbine, step_down=step_down)



end = False
time_list = []
s_l = []
v_l = []
p_l = []


def print_stuff_tank_nozzle():
    print(round(clock, 1), "s,",
          "Tank: ",
          round(tank.p0, 0), "Pa, ",
          round(tank.T0, 0), "K, ",
          round(tank.d0, 1), "kg/m^3, ",
          round(1000 * tank.m(), 1), "g, ",
          "Nozzle: ",
          round(nozzle.mdot_max(), 5), "kg/s, ",
          #round(test_nozzle.d0, 3), "kg/m^3, ",
          round(nozzle.q_max(), 6), "m^3/s, ",
          round(nozzle.p0 / nozzle.pb, 2), "pressure ratio, ",
          round(nozzle.M_exhaust(), 2), "Mach, ",
          round(1000 * nozzle.exhaust_diameter(), 5), "mm ", )

def print_stuff_drive():
    print(round(clock, 1), "s,",
          "Vehicle data: ",
          round(car.frictional_torque(), 4), "Nm frictional",
          round(car.s, 2), "m,",
          round(car.v, 2), "m/s,",
          round(car.a, 2), "m/s^2,",
          round(car.t, 3), "Nm net, ",
          round(car.rpm, 1), "rpm, ",
          )

def print_stuff_turbine():
    print(round(clock, 1), "s,",
          round(turbine.p0, 1), "Pa (inlet), ",
          round(turbine.max_power(), 2), "W max, ",
          round(car.rpm * step_down, 0), "rpm operation, ",
          round(turbine.no_load_rpm(), 0), "rpm max, ",
          round(turbine.torque(car.rpm * step_down), 3), "Nm operation, ",
          turbine.specific_speed(car.rpm * step_down * 2 * np.pi / 60), "specific speed"
          )


while not end:
    car = car.update(time_step, turbine)
    turbine = TurbineDrive(p0=nozzle.pb, T0=nozzle.T0, d0=nozzle.d0, p1=100000, T1=298, d1=1, nozzle=nozzle)
    nozzle = Nozzle(p0=tank.p0, T0=tank.T0, d0=tank.d0, pb=0.995*tank.p0)
    if tank.p0 > 100000/0.995:
        tank = tank.update(time_step, nozzle)
    else:
        tank = Tank(p0=100000, T0=298, d0=1)

    #print_stuff_drive()
    #print_stuff_tank_nozzle()
    #print_stuff_turbine()

    time_list.append(clock)
    s_l.append(car.s)
    v_l.append(car.v)
    p_l.append(tank.p0)
    if car.s > 25 or clock > 30:
        end = True
    clock += time_step
#'''



'''
tank = Tank(p0=650000, T0=298, d0=7)
nozzle = Nozzle(p0=tank.p0, T0=tank.T0, d0=tank.d0, pb=0.995 * tank.p0)
turbine = TurbineDrive(p0=nozzle.pb, T0=nozzle.T0, d0=nozzle.d0, p1=p_ambient, T1=298, d1=1, nozzle=nozzle)
print(turbine.max_power())
'''

'''
plot.plot(time_list, p_l)
plot.title('Tank pressure time history')
plot.xlabel('time (s)')
plot.ylabel('pressure (Pa)')
plot.show()
'''

plot.plot(time_list, s_l)
plot.title('Position time history')
plot.xlabel('time (s)')
plot.ylabel('distance (m)')
plot.show()

plot.plot(time_list, v_l)
plot.title('Velocity time history')
plot.xlabel('time (s)')
plot.ylabel('velocity (m/s)')
plot.show()
#'''
