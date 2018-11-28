import numpy as np
from numpy.linalg import inv as linv

def rotMat(theta):
    return np.array([[np.cos(theta), -np.sin(theta),         0.],
                     [np.sin(theta),  np.cos(theta),         0.],
                     [           0.,             0.,         1.]])

def _norm_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))

def mul_gaussians(g1, g2):
    v1inv = linv(g1[1])
    v2inv = linv(g2[1])

    var_out = linv(v1inv + v2inv)
    mu_out  = linv(var_out).dot(v1inv.dot(g1[0]) + v2inv.dot(g2[0]))
    return (mu_out, var_out)
     

