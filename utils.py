from collections import namedtuple
import numpy as np
import math
from numpy.linalg import inv as linv

observationTuple = namedtuple('observationTuple', ['height', 'bearing', 'beacon_id', 'command', 'dt'])

def rotMat(theta):
    return np.array([[np.cos(theta), -np.sin(theta),         0.],
                     [np.sin(theta),  np.cos(theta),         0.],
                     [           0.,             0.,         1.]])

def mul_gaussians(g1, g2):
    v1inv = linv(g1[1])
    v2inv = linv(g2[1])

    var_out = linv(v1inv + v2inv)
    mu_out  = linv(var_out).dot(v1inv.dot(g1[0]) + v2inv.dot(g2[0]))
    return (mu_out, var_out)

def mul_gaussians_wrong(g1, g2):
    v1inv = linv(g1[1])
    v2inv = linv(g2[1])

    var_out = linv(v1inv + v2inv)
    mu_out  = g1[0] + g1[1].dot(linv(g1[1] + g2[1])).dot(g2[0]-g1[0])
    return (mu_out, var_out)

def preprocess_data(filename):
    with open(filename, 'r') as f:
        data = f.read().split('\n')[1:-1]
        data = [d.split(', ') for d in data][:1000]
        last_obs_idx = len(data)-1
        for i, d in reversed(list(zip(range(len(data)), data))):
            if d[0] != '-1000':
                last_obs_idx = i
                break
    def sanitize_data_point(d):
        d[0] = float(d[0]) if d[0] != '-1000' else None
        d[1] = float(d[1]) if d[1] != '-1000' else None
        d[2] = int(d[2]) if d[2] != '-1000' else None
        d[3] = int(d[3])
        d[4] = float(d[4])
        return d

    data = [sanitize_data_point(d) for d in data[:last_obs_idx+1]]
    data = [observationTuple(height=d[0], 
                             bearing=d[1], 
                             beacon_id=d[2], 
                             command=d[3], 
                             dt=d[4]) for d in data]
    return data

def print_params(u):
    print("=== Mean: {:8.3f}, {:8.3f}, {:8.3f}".format(*u[0]))
    print("=== Covariance ===")
    print("{:8.3f}, {:8.3f}, {:8.3f}".format(*u[1][0]))
    print("{:8.3f}, {:8.3f}, {:8.3f}".format(*u[1][1]))
    print("{:8.3f}, {:8.3f}, {:8.3f}".format(*u[1][2]))
