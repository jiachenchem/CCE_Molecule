import numpy
from cmath import sqrt as csqrt
from math import *

# read xyz file ========================================
def read_xyz(name):
    """ given file name, read xyz file
    return element and position list"""
    f = open(name, 'r')
    N_atom = int(f.readline())
    f.readline()
    elem = []
    pos = []
    lines = f.readlines()
    count = 0
    for line in lines:
        temp = line.split()
        if temp[0] == 'H' or (temp[0] == 'V' and count == 0):
            elem.append(temp[0])
            pos.append( [float(temp[1]), float(temp[2]), float(temp[3])] )
        count += 1
    pos = numpy.array(pos)
    return elem, pos
# =====================================================

# dipole-dipole interactio based on a xyz file =========
def dd_int(pos):
    """dipole-dipole interaction from molecular structure 
    For proton-proton 120.096 KHz / 1.A;
    for proton-electron, 79.164 MHz / 1.A
    Malcolm Levitt second edition P212"""
    
    # count number of spins, for this case V and H
    Nu_spin = elem.count('H')
    print('number of nuclear spins:', Nu_spin)
    # matrix for dipole-dipole interaction, the first one is electron spin, positioned at metal
    indices_H = [i for i, x in enumerate(elem) if x == "H"]
    indices_V = [i for i, x in enumerate(elem) if x == "V"]
    print('index H', indices_H)
    dd = numpy.zeros((Nu_spin+1, Nu_spin+1))
    for i in range(Nu_spin+1):
        for j in range(i+1, Nu_spin+1):
            if i == 0: # electron spin and metal
                vec = pos[indices_V[0], :] - pos[indices_H[j-1], :] 
                dist = numpy.linalg.norm( vec )
                #print('dist', dist)
                theta = acos(vec[2]/dist)
                #dd[i,j] = 1.0 * 497.404E6 / 1.7608E11 / (dist**3)
                dd[i,j] += (3 * cos(theta)**2 - 1.0) / 2.0 * 497.404E6 / 1.7608E11 / (dist**3)
            else:
                vec = pos[indices_H[i-1], :] - pos[indices_H[j-1], :]
                dist = numpy.linalg.norm( vec )
                theta = acos(vec[2]/dist)
                #dd[i,j] = 1.0 * 754.585E3 / 1.7608E11 / (dist**3)
                dd[i,j] += (3 * cos(theta)**2 - 1.0) / 2.0 * 754.585E3 / 1.7608E11 / (dist**3)
    return dd
                   

#=========================================================
