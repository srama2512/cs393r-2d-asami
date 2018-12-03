from collections import namedtuple
import numpy as np
import math
from numpy.linalg import inv as linv

observationTuple = namedtuple('observationTuple', ['height', 'bearing', 'beacon_id', 'command', 'dt', 'dist'])

def rotMat(theta):
    return np.array([[np.cos(theta), -np.sin(theta),         0.],
                     [np.sin(theta),  np.cos(theta),         0.],
                     [           0.,             0.,         1.]])

def mul_gaussians(g1, g2):
    v1inv = linv(g1[1])
    v2inv = linv(g2[1])

    var_out = linv(v1inv + v2inv)
    mu_out  = var_out.dot(v1inv.dot(g1[0]) + v2inv.dot(g2[0]))
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
        data = [d.split(', ') for d in data]
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
        d[5] = float(d[5]) if d[5] != '-1000' else None
        return d

    data = [sanitize_data_point(d) for d in data[:last_obs_idx+1]]
    data = [observationTuple(height=d[0], 
                             bearing=d[1], 
                             beacon_id=d[2], 
                             command=d[3], 
                             dt=d[4],
                             dist=d[5]) for d in data]
    return data

def process_gt_data(filename):
	with open(filename, 'r') as f:
		data = f.read().split('\n')[1:-1]
		data = [[float(i) for i in d.split(', ')] for d in data]
		return np.array([[0., 0., 0.]] + data)

def compute_action_model(gt_data, cmds):
	gt_deltas_map = dict()
	for i in range(40):
		gt_deltas_map[i] = []
	for i, cmd in enumerate(cmds):
		delta = rotMat(-gt_data[i,2]).dot(gt_data[i+1]-gt_data[i])
		_norm_angle = lambda t: math.atan2(math.sin(t), math.cos(t))
		delta[2] = _norm_angle(delta[2])
		delta = delta*30.
		gt_deltas_map[cmd].append(delta)

	gt_action_model = {'mean': np.zeros((40, 3)), 'cov': np.zeros((40, 3, 3))}
	for i in range(40):
		gt_action_model['mean'][i] = np.mean(np.array(gt_deltas_map[i]), axis=0)
		gt_action_model['cov'][i]  = np.cov(np.array(gt_deltas_map[i]), rowvar=False)
	return gt_action_model

def print_params(u):
    print("=== Mean: {:8.3f}, {:8.3f}, {:8.3f}".format(*u[0]))
    print("=== Covariance ===")
    print("{:8.3f}, {:8.3f}, {:8.3f}".format(*u[1][0]))
    print("{:8.3f}, {:8.3f}, {:8.3f}".format(*u[1][1]))
    print("{:8.3f}, {:8.3f}, {:8.3f}".format(*u[1][2]))
