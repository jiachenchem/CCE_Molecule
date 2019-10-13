#! /usr/bin/env python3
"""cluster correlation expansion for central spin decoherence
   PRB 74, 195301, 2006
   New J. Phys.9 226, 2007
   PRB 78, 085315, 2008"""

import sys
import random
import numpy
from math import *
from tools import *
from evo import *
import matplotlib.pyplot as plt 

Bp = 1.0 
Be = 658.169 * Bp
# about unit of time ==========================
# for proton H = omega_0 Sz
# for B = 1T, omega_0 = 42.578E6 s^-1
# Energy unit is 42.578E6 Hz
# so units for time is 2.3486E-08 s
#==============================================
TIME_UNIT = 1.0 / 42.578E6

Nt = 2000
dt = 10.
L_out = numpy.zeros(Nt)
Time = numpy.zeros(Nt)
for i in range(Nt):
    Time[i] += i * dt * TIME_UNIT 

N_orien = 1 # number of orientation
N_init = 10 # number of random initial configuration

for i in range(N_orien):
    filename = './orientations/' + str(i)+'.xyz'
    print('strucutre from file', filename)
    ele, pos = read_xyz(filename)
    N_bath = len(ele) - 1
    print('number of bath:', N_bath)
#== generate initial configuration ========================
    iconf = []
    for i in range(N_init):
        temp = []
        for j in range(N_bath):
            if random.random() > 0.5:
                temp.append('0')
            else:
                temp.append('1')
        iconf.append(''.join(temp))
    #print(iconf[0])
#===========================================================
    N_pair = N_bath * (N_bath-1) /2
    L = Hahn_12(N_bath, iconf, Nt, dt, Bp, pos) 
    for j in range(Nt):
        for k in range(N_init):
            L_out[j] += numpy.absolute(L[j,k])

f=open('L_Hahn.out','w')
for i in range(Nt):
    f.write( format(Time[i], '20.8f') + format(L_out[i], '20.8f') + "\n" )
f.close()
