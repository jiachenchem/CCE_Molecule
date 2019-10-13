from math import *
import numpy
from qutip import *
# Sigma operators  ==============================
sgm_x = numpy.array([[0.0, 1.0], [1.0, 0.0]])
sgm_y = numpy.array([[0.0, -1j], [1j, 0.0]])
sgm_z = numpy.array([[1.0, 0.0], [0.0, -1.0]])
sgm_p = numpy.array([[0.0, 1.0], [0.0,  0.0]])
sgm_n = numpy.array([[0.0, 0.0], [1.0,  0.0]])
ident = numpy.array([[1.0, 0.0], [0.0,  1.0]])
#================================================

# two spin hamiltonian ==========================
def two_H(vec1, vec2, Bp):
    """ Hamiltonian of two spin 1/2  """

    Be = 658.169 * Bp
    H_int = numpy.zeros((4,4), complex) # intrinsic spin dipole-dipole interaction
    H_ext = numpy.zeros((4,4), complex) # extrinsic interaction mediated by electron
    Zeeman = numpy.kron(sgm_z, ident) + numpy.kron(ident, sgm_z) # B.Jz
    diag = numpy.kron(sgm_z, sgm_z) # Jz^i.Jz^j
    off_diag = numpy.kron(sgm_p, sgm_n) + numpy.kron(sgm_n, sgm_p) # J+^i.J-^j + J-^i.J+^j
    
    vec = vec1 - vec2
    dist = numpy.linalg.norm(vec)
    theta = acos(vec[2]/dist)
    dd = (3 * cos(theta)**2 - 1.0) / 2.0 * 120.096E3 / 42.578E6 / (dist**3)  
    H_int += Bp * Zeeman 
    H_int += dd *  2.0 * diag 
    H_int +=  -0.5 * dd *  off_diag  

    dist = numpy.linalg.norm(vec1)
    theta = acos(vec1[2]/dist)
    a1 = (3 * cos(theta)**2 - 1.0) / 2.0 * 79.164E6 / 42.578E6 / (dist**3)
    dist = numpy.linalg.norm(vec2)
    theta = acos(vec2[2]/dist)
    a2 = (3 * cos(theta)**2 - 1.0) / 2.0 * 79.164E6 / 42.578E6 / (dist**3)
    H_ext += a1 * 2. * numpy.kron(sgm_z, ident) + a2 * 2. * numpy.kron(ident, sgm_z)
    H_ext += a1 * a2 / Be / 4.0  * off_diag # not sure about a factor of 2

    return H_int + H_ext, H_int - H_ext
#===================================================
def twelve_H(pos, Bp):
    """ Hamiltonian of 12 spin 1/2 """
    #from qutip import *

    Be = 658.169 * Bp

    N_spin = 12
    N_Hil = 2**N_spin
    H_int = numpy.zeros((N_Hil,N_Hil), complex) # intrinsic spin dipole-dipole interaction
    H_ext = numpy.zeros((N_Hil,N_Hil), complex) # extrinsic interaction mediated by electron

    Zeeman = numpy.zeros((N_Hil,N_Hil), complex)
    ext_Zeeman = numpy.zeros((N_Hil,N_Hil), complex)
    for i in range(N_spin):
        vec1 = pos[i+1] - pos[0]
        dist = numpy.linalg.norm(vec1)
        theta = acos(vec1[2]/dist)
        a1 = (3 * cos(theta)**2 - 1.0) / 2.0 * 79.164E6 / 42.578E6 / (dist**3)
        if i == 0:
            mat = tensor(sigmaz(), identity(2**(N_spin-1))) 
        elif i == N_spin:
            mat = tensor(identity(2**(N_spin-1)), sigmaz())
        else:
            mat = tensor(identity(2**i), sigmaz(), identity(2**(N_spin-i-1)))
        Zeeman += numpy.array(mat)
        ext_Zeeman += a1 * 2. * numpy.array(mat)

    H_int += Bp * Zeeman
    for i in range(1, N_spin):
        for j in range(i+1, N_spin+1):
            vec1 = pos[i] - pos[0]
            vec2 = pos[j] - pos[0]
            vec = vec1 - vec2
            dist = numpy.linalg.norm(vec)
            theta = acos(vec[2]/dist)
            dd = (3 * cos(theta)**2 - 1.0) / 2.0 * 120.096E3 / 42.578E6 / (dist**3)
            # general expression
            #tensor( identity(2**(i-1)), sigmaz(), identity(2**(j-i-1)), sigmaz(), identity(2**(N_spin+1-j-1)) )
            if i-1 > 0 and j-i-1 > 0 and N_spin-j > 0:
                H_int += 2.0*dd*numpy.array(tensor(identity(2**(i-1)),sigmaz(),identity(2**(j-i-1)),sigmaz(),identity(2**(N_spin-j))))
                H_int += -0.5*dd*numpy.array(tensor(identity(2**(i-1)),sigmam(),identity(2**(j-i-1)),sigmap(),identity(2**(N_spin-j))))
                H_int += -0.5*dd*numpy.array(tensor(identity(2**(i-1)),sigmap(),identity(2**(j-i-1)),sigmam(),identity(2**(N_spin-j))))
            elif i-1 == 0 and j-i-1 > 0 and N_spin-j > 0:
                H_int += 2.0*dd*numpy.array(tensor(sigmaz(),identity(2**(j-i-1)),sigmaz(),identity(2**(N_spin-j))))
                H_int += -0.5*dd*numpy.array(tensor(sigmam(),identity(2**(j-i-1)),sigmap(),identity(2**(N_spin-j))))
                H_int += -0.5*dd*numpy.array(tensor(sigmap(),identity(2**(j-i-1)),sigmam(),identity(2**(N_spin-j))))
            elif i-1 > 0 and j-i-1 == 0 and N_spin-j > 0:
                H_int += 2.0*dd*numpy.array(tensor(identity(2**(i-1)),sigmaz(),sigmaz(),identity(2**(N_spin-j))))
                H_int += -0.5*dd*numpy.array(tensor(identity(2**(i-1)),sigmam(),sigmap(),identity(2**(N_spin-j))))
                H_int += -0.5*dd*numpy.array(tensor(identity(2**(i-1)),sigmap(),sigmam(),identity(2**(N_spin-j))))
            elif i-1 > 0 and j-i-1 > 0 and N_spin-j == 0:
                H_int += 2.0*dd*numpy.array(tensor(identity(2**(i-1)),sigmaz(),identity(2**(j-i-1)),sigmaz()))
                H_int += -0.5*dd*numpy.array(tensor(identity(2**(i-1)),sigmam(),identity(2**(j-i-1)),sigmap()))
                H_int += -0.5*dd*numpy.array(tensor(identity(2**(i-1)),sigmap(),identity(2**(j-i-1)),sigmam()))
            elif i-1 == 0 and j-i-1 == 0 and N_spin-j > 0:
                H_int += 2.0*dd*numpy.array(tensor(sigmaz(),sigmaz(),identity(2**(N_spin-j))))
                H_int += -0.5*dd*numpy.array(tensor(sigmam(),sigmap(),identity(2**(N_spin-j))))
                H_int += -0.5*dd*numpy.array(tensor(sigmap(),sigmam(),identity(2**(N_spin-j))))
            elif i-1 == 0 and j-i-1 > 0 and N_spin-j == 0:
                H_int += 2.0*dd*numpy.array(tensor(sigmaz(),identity(2**(j-i-1)),sigmaz()))
                H_int += -0.5*dd*numpy.array(tensor(sigmam(),identity(2**(j-i-1)),sigmap()))
                H_int += -0.5*dd*numpy.array(tensor(sigmap(),identity(2**(j-i-1)),sigmam()))
            elif i-1 > 0 and j-i-1 == 0 and N_spin-j == 0:
                H_int += 2.0*dd*numpy.array(tensor(identity(2**(i-1)),sigmaz(),sigmaz()))
                H_int += -0.5*dd*numpy.array(tensor(identity(2**(i-1)),sigmam(),sigmap()))
                H_int += -0.5*dd*numpy.array(tensor(identity(2**(i-1)),sigmap(),sigmam()))
            else:
                raise Exception('An error occurred in two-body interaction')

            dist = numpy.linalg.norm(vec1)
            theta = acos(vec1[2]/dist)
            a1 = (3 * cos(theta)**2 - 1.0) / 2.0 * 79.164E6 / 42.578E6 / (dist**3)
            dist = numpy.linalg.norm(vec2)
            theta = acos(vec2[2]/dist)
            a2 = (3 * cos(theta)**2 - 1.0) / 2.0 * 79.164E6 / 42.578E6 / (dist**3)

            if i-1 > 0 and j-i-1 > 0 and N_spin-j > 0:
                H_ext +=  a1*a2/Be/4.0*numpy.array(tensor(identity(2**(i-1)),sigmam(),identity(2**(j-i-1)),sigmap(),identity(2**(N_spin-j))))
                H_ext +=  a1*a2/Be/4.0*numpy.array(tensor(identity(2**(i-1)),sigmap(),identity(2**(j-i-1)),sigmam(),identity(2**(N_spin-j))))
            elif i-1 == 0 and j-i-1 > 0 and N_spin-j > 0:
                H_ext += a1*a2/Be/4.0*numpy.array(tensor(sigmam(),identity(2**(j-i-1)),sigmap(),identity(2**(N_spin-j))))
                H_ext += a1*a2/Be/4.0*numpy.array(tensor(sigmap(),identity(2**(j-i-1)),sigmam(),identity(2**(N_spin-j))))
            elif i-1 > 0 and j-i-1 == 0 and N_spin-j > 0:
                H_ext += a1*a2/Be/4.0*numpy.array(tensor(identity(2**(i-1)),sigmam(),sigmap(),identity(2**(N_spin-j))))
                H_ext += a1*a2/Be/4.0*numpy.array(tensor(identity(2**(i-1)),sigmap(),sigmam(),identity(2**(N_spin-j))))
            elif i-1 > 0 and j-i-1 > 0 and N_spin-j == 0:
                H_ext += a1*a2/Be/4.0*numpy.array(tensor(identity(2**(i-1)),sigmam(),identity(2**(j-i-1)),sigmap()))
                H_ext += a1*a2/Be/4.0*numpy.array(tensor(identity(2**(i-1)),sigmap(),identity(2**(j-i-1)),sigmam()))
            elif i-1 == 0 and j-i-1 == 0 and N_spin-j > 0:
                H_ext += a1*a2/Be/4.0*numpy.array(tensor(sigmam(),sigmap(),identity(2**(N_spin-j))))
                H_ext += a1*a2/Be/4.0*numpy.array(tensor(sigmap(),sigmam(),identity(2**(N_spin-j))))
            elif i-1 == 0 and j-i-1 > 0 and N_spin-j == 0:
                H_ext += a1*a2/Be/4.0*numpy.array(tensor(sigmam(),identity(2**(j-i-1)),sigmap()))
                H_ext += a1*a2/Be/4.0*numpy.array(tensor(sigmap(),identity(2**(j-i-1)),sigmam()))
            elif i-1 > 0 and j-i-1 == 0 and N_spin-j == 0:
                H_ext += a1*a2/Be/4.0*numpy.array(tensor(identity(2**(i-1)),sigmam(),sigmap()))
                H_ext += a1*a2/Be/4.0*numpy.array(tensor(identity(2**(i-1)),sigmap(),sigmam()))
            else:
                raise Exception('An error occurred in two-body interaction')
    H_ext += ext_Zeeman
    A = Qobj(H_int + H_ext)
    print(A.isherm)
    B = Qobj(H_int - H_ext)
    print(B.isherm)
    C = Qobj(H_int)
    print(C.isherm)
    D = Qobj(H_ext)
    print(D.isherm)
    return H_int + H_ext, H_int - H_ext
#========================================================
