from filterpy.kalman import ExtendedKalmanFilter, InformationFilter
from numpy.linalg import norm as Lnorm
from collections import namedtuple
from utils import rotMat

import numpy as np
import math

class EKFforward(ExtendedKalmanFilter):
    def predict_x(self, u=0):
        j13 = -np.sin(self.x[2])*u[0] + np.cos(self.x[2])*u[1]
        j23 = -np.cos(self.x[2])*u[0] - np.sin(self.x[2])*u[1]

        self.F = np.array([[1., 0., j13], 
                           [0., 1., j23],
                           [0., 0., 1. ]])
        self.x = self.x + rotMat(self.x[2]).dot(u)

observationTuple = namedtuple('observationTuple', ['height', 'bearing', 'beacon_id', 'command', 'dt'])

def _norm_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))

class PolynomialRegression(object):
    # TODO
    def __init__(self):
        pass

class ActionMapper(object):
    # TODO
    def __init__(self):
        pass

class EM(object):
    def __init__(self, n_state=3, n_action=3, n_meas=2):
        self.n_state = n_state
        self.n_action = n_action
        self.n_meas = n_meas

        self.forward_model = EKFforward(dim_x=n_state, dim_z=n_meas)
        self.backward_model = InformationFilter(***)

        self.sensor_model = PolynomialRegression(d=3)
        self.action_model = ActionMapper(cmd_size, n_action=n_action)

        self._initialize_models()

        self.bpos = {} # TODO map from bid to (x, y)

    def _set_priors(self):
        self.forward_model.x = np.array([0., 0., 0.])
        self.forward_model.P = np.diag([1000., 1000., np.pi / 10.]) # belief covariance
        self.alphas = []
        self.betas = []
        self.gammas = []
        self.delta = [] # b * beta

    def _initialize_models(self):
        self.forward_model.Q = np.diag([10., 10., 0.1]) # action model
        self.forward_model.R = np.diag([10., 0.2])  # sensor model

    def hx_f(self, b):
        # x - current state, b - beacon position
        def hx(x):
            dist = Lnorm(x[:2] - b[:2])
            bear = _norm_angle(math.atan2(b[1]-x[1], b[0]-x[0]) - x[2])
            ot = np.array([self.sensor_model.predict([dist]), bear])
            return ot
        return hx

    def HJacobian_f_at(self, b):
        # x - current state, b - beacon position
        def HJacobian_at(x):
            dx = x[:2] - b[:2]
            dist = Lnorm(dx)
            return [[dx[0]/dist    , dx[1]/dist   , 0 ],
                    [-dx[1]/dist**2, dx[0]/dist**2, -1]]
        return HJacobian_at

    def Estep(self, data):
        # data - list of observationTuples
        # Note: height, bearing are after taking command for dt
        self._set_priors()
        # ==== forward model prediction ====
        for data_t in data:
            ht, bear, bid, cmd, dt = data_t
            act = self.action_model.get(cmd) * dt
            HJacobian_at = self.HJacobian_f_at(self.bpos[bid])
            hx = self.hx_f(self.bpos[bid])
            if ht is not None:
                obs = np.array([ht, bear])
            else:
                obs = None

            self.forward_model.predict(u=act)
            self.forward_model.update(obs, HJacobian_at, hx)

            self.alphas.append(self.forward_model.x_post)

