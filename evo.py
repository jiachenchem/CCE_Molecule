# -*- coding: utf-8 -*-
"""
coherence function due-to pair-wise contribution 
from random initial configuration

@author: Jia
"""

import sys
import random
import numpy
from math import *
from tools import *
from Ham import *
from cmath import exp as cexp

def FID_2(N_bath, iconf, Nt, dt, Bp, pos):
    N_init = len(iconf)
    L = numpy.ones((Nt, N_init), complex)
    for i in range(1, N_bath):
        print('pair i:',i)
        for j in range(i+1, N_bath+1):
            vec1 = pos[i] - pos[0]
            vec2 = pos[j] - pos[0]
            H1, H2 = two_H(vec1, vec2, Bp)
            w1, v1 = numpy.linalg.eigh(H1)
            w2, v2 = numpy.linalg.eigh(H2)
            #print('check if Hamiltonians for spins are the same:', numpy.array_equal(v1, v2))
            for en in range(N_init):
                j0 = int(str(iconf[en][i-1])+str(iconf[en][j-1]), 2) # inital state as a two-spin basis function
                j1 = [ numpy.conj(v1[j0,k]) for k in range(4) ] # initial state in the basis of eigenstats
                j2 = [ numpy.conj(v2[j0,k]) for k in range(4) ]

                T = 0.0
                for k in range(Nt):
                    jt1 = numpy.zeros(4,complex)
                    jt2 = numpy.zeros(4,complex)
                    for l in range(4):
                        jt1[l] = j1[l] * cexp(-1j * w1[l] * T)
                        jt2[l] = j2[l] * cexp(-1j * w2[l] * T)
                    capb1 = numpy.einsum('ij,i->j', v1.transpose(), jt1) # change to computational basis
                    capb2 = numpy.einsum('ij,i->j', v2.transpose(), jt2)
                    L[k,en] = L[k,en] * numpy.dot(numpy.conj(capb1), capb2) #* cexp( -1j * fac * T)
                    T += dt
    return L
#===========================================================================================================

#===========================================================================================================
def Hahn_2(N_bath, iconf, Nt, dt, Bp, pos):
    """repeating code here, needs to be condensed"""
    N_init = len(iconf)
    L = numpy.ones((Nt, N_init), complex)
    for i in range(1, N_bath):
        print('pair i:',i)
        for j in range(i+1, N_bath+1):
            vec1 = pos[i] - pos[0]
            vec2 = pos[j] - pos[0]
            H1, H2 = two_H(vec1, vec2, Bp)
            w1, v1 = numpy.linalg.eigh(H1)
            w2, v2 = numpy.linalg.eigh(H2)
            #print('check if Hamiltonians for spins are the same:', numpy.array_equal(v1, v2))
            for en in range(N_init):
                j0 = int(str(iconf[en][i-1])+str(iconf[en][j-1]), 2) # inital state as a two-spin basis function
                #print('j0', j0)
                j_up = [ numpy.conj(v1[j0,k]) for k in range(4) ] # initial state in the basis of eigenstats
                j_dn = [ numpy.conj(v2[j0,k]) for k in range(4) ]

                #j00 = numpy.array([0.0, 0.0, 0.0, 0.0])
                #for i in range(4):
                #    if i == j0:
                #        j00[i] = 1.0
                #print(j00)
                #tt1 = numpy.einsum('ij,i->j', v1, j00)
                #tt2 = numpy.einsum('ij,i->j', v2, j00)
                #print(numpy.array_equal(j_up, tt1))
                #print(numpy.array_equal(j_dn, tt2))
                #print(j_up)
                #print(tt1)
                #sys.exit()

                T = 0.0
                for k in range(Nt):
                    jt_up = numpy.zeros(4,complex)
                    jt_up_2 = numpy.zeros(4,complex)
                    jt_up_3 = numpy.zeros(4,complex)
                    jt_dn = numpy.zeros(4,complex)
                    jt_dn_2 = numpy.zeros(4,complex)
                    jt_dn_3 = numpy.zeros(4,complex)

                    for l in range(4): 
                        jt_up[l] = j_up[l] * cexp(-1j * w1[l] * T)
                        jt_dn[l] = j_dn[l] * cexp(-1j * w2[l] * T)

                    tt = numpy.einsum('ij,i->j', v1.transpose(), jt_up) # back to computational basis
                    jt_up_2 = numpy.einsum('ij,i->j', v2, tt) # go to v2 basis
                    tt = numpy.einsum('ij,i->j', v2.transpose(), jt_dn)
                    jt_dn_2 = numpy.einsum('ij,i->j', v1, tt)

                    for l in range(4):
                        jt_up_3[l] = jt_up_2[l] * cexp(-1j * w2[l] * T)
                        jt_dn_3[l] = jt_dn_2[l] * cexp(-1j * w1[l] * T)

                    capb1 = numpy.einsum('ij,i->j', v2.transpose(), jt_up_3) # change to computational basis
                    capb2 = numpy.einsum('ij,i->j', v1.transpose(), jt_dn_3)
                    L[k,en] = L[k,en] * numpy.dot(numpy.conj(capb1), capb2) #* cexp( -1j * fac * T)
                    T += dt
    return L
#===========================================================================================================

def FID_12(N_bath, iconf, Nt, dt, Bp,  pos):
    N_init = len(iconf)
    L = numpy.ones((Nt, N_init), complex)
    H1, H2 = twelve_H(pos, Bp)
    w1, v1 = numpy.linalg.eigh(H1)
    w2, v2 = numpy.linalg.eigh(H2)

    for en in range(N_init):
        print('initial configuration:', en)
        j0 = int(str(iconf[en]), 2) # inital state as a two-spin basis function
        print('j0', j0)
        j1 = [ numpy.conj(v1[j0,k]) for k in range(2**N_bath) ] # initial state in the basis of eigenstats, state projection
        j2 = [ numpy.conj(v2[j0,k]) for k in range(2**N_bath) ]

        T = 0.0
        for k in range(Nt):
            jt1 = numpy.zeros(2**N_bath,complex)
            jt2 = numpy.zeros(2**N_bath,complex)
            for l in range(2**N_bath):
                jt1[l] = j1[l] * cexp(-1j * w1[l] * T)
                jt2[l] = j2[l] * cexp(-1j * w2[l] * T)
            capb1 = numpy.einsum('ij,i->j', v1.transpose(), jt1) # change to computational basis
            capb2 = numpy.einsum('ij,i->j', v2.transpose(), jt2)
            L[k,en] = L[k,en] * numpy.dot(numpy.conj(capb1), capb2) #* cexp( -1j * fac * T)
            T += dt
    return L
#========================================================================================

def Hahn_12(N_bath, iconf, Nt, dt, Bp,  pos):
    N_init = len(iconf)
    L = numpy.ones((Nt, N_init), complex)
    H1, H2 = twelve_H(pos, Bp)
    w1, v1 = numpy.linalg.eigh(H1)
    w2, v2 = numpy.linalg.eigh(H2)

    for en in range(N_init):
        print('initial configuration:', en)
        j0 = int(str(iconf[en]), 2) # inital state as a two-spin basis function
        print('j0', j0)
        j_up = [ numpy.conj(v1[j0,k]) for k in range(2**N_bath) ] # initial state in the basis of eigenstats, state projection
        j_dn = [ numpy.conj(v2[j0,k]) for k in range(2**N_bath) ]

        T = 0.0
        for k in range(Nt):
            print('T', T)
            jt_up = numpy.zeros(2**N_bath, complex)
            jt_up_2 = numpy.zeros(2**N_bath, complex)
            jt_up_3 = numpy.zeros(2**N_bath, complex)
            jt_dn = numpy.zeros(2**N_bath, complex)
            jt_dn_2 = numpy.zeros(2**N_bath, complex)
            jt_dn_3 = numpy.zeros(2**N_bath, complex)

            for l in range(2**N_bath):
                jt_up[l] = j_up[l] * cexp(-1j * w1[l] * T)
                jt_dn[l] = j_dn[l] * cexp(-1j * w2[l] * T)

            tt = numpy.einsum('ij,i->j', v1.transpose(), jt_up) # back to computational basis
            jt_up_2 = numpy.einsum('ij,i->j', v2, tt) # go to v2 basis
            tt = numpy.einsum('ij,i->j', v2.transpose(), jt_dn)
            jt_dn_2 = numpy.einsum('ij,i->j', v1, tt)

            for l in range(2**N_bath):
                jt_up_3[l] = jt_up_2[l] * cexp(-1j * w2[l] * T)
                jt_dn_3[l] = jt_dn_2[l] * cexp(-1j * w1[l] * T)

            capb1 = numpy.einsum('ij,i->j', v2.transpose(), jt_up_3) # change to computational basis
            capb2 = numpy.einsum('ij,i->j', v1.transpose(), jt_dn_3)
            L[k,en] = L[k,en] * numpy.dot(numpy.conj(capb1), capb2) # cexp( -1j * fac * T)
            T += dt
    return L
#==============================================================================================
