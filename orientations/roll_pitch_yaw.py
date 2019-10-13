# -*- coding: utf-8 -*-
"""
Random orientation of a molecule via roll-pitch-yaw angles
James J. Kuffner "Effective Sampling and Distance Metrics for 3D Rigid Body Path Planning
"
In Proc. 2004 IEEE Int’l Conf. on Robotics and Automation (ICRA 2004)
@author: Jia
"""
import sys
import random
import numpy
from math import *
from ase.io import read, write

N = 1000 # number of orientations to generate
for i in range(N):
    origin = read('vc1.xyz')
    tt = origin.get_center_of_mass()
    origin.translate(-1.0*tt)
    theta = 2.0 * pi * random.random()  - pi
    phi = acos(1.0 - 2.0 * random.random()) + pi / 2.0
    if random.random() < 0.5:
        if phi < pi:
            phi = phi + pi
        else:
            phi = phi - pi
    eta = 2 * pi * random.random() - pi
    origin.rotate(theta * 180. / pi, 'x')
    origin.rotate(phi * 180. / pi, 'y')
    origin.rotate(eta * 180. / pi, 'z')
    filename = str(i) + '.xyz'
    print(filename)
    origin.write(filename)