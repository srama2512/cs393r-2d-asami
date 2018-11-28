from filterpy.kalman import ExtendedKalmanFilter
from information_filter import BackwardInformationFilter
from numpy.linalg import norm as Lnorm
from collections import namedtuple
from utils import rotMat, _norm_angle
from sklearn.linear_model import LinearRegression

import numpy as np
import math

FIELD_X = 3500
FIELD_X = 2000
GRASS_Y = 4000
GRASS_X = 3000

HALF_FIELD_Y = FIELD_Y/2.0 // 1300
HALF_FIELD_X = FIELD_X/2.0 // 1000
HALF_GRASS_Y = GRASS_Y/2.0
HALF_GRASS_X = GRASS_X/2.0

BEACONS = [(-HALF_FIELD_X, -HALF_FIELD_Y),       #WO_BEACON_BLUE_YELLOW
           (-HALF_FIELD_X, -HALF_FIELD_Y),       #WO_BEACON_YELLOW_BLUE,
           (-HALF_FIELD_X/2, HALF_FIELD_Y),      #WO_BEACON_BLUE_PINK
           (-HALF_FIELD_X/2, HALF_FIELD_Y),      #WO_BEACON_PINK_BLUE
           (0, -HALF_FIELD_Y),                   #WO_BEACON_PINK_YELLOW
           (0, -HALF_FIELD_Y)]                   #WO_BEACON_YELLOW_PINK,


class EKFforward(ExtendedKalmanFilter):
    def predict_x(self, u=0):
        j13 = -np.sin(self.x[2])*u[0] + np.cos(self.x[2])*u[1]
        j23 = -np.cos(self.x[2])*u[0] - np.sin(self.x[2])*u[1]

        self.F = np.array([[1., 0., j13], 
                           [0., 1., j23],
                           [0., 0., 1. ]])
        self.x = self.x + rotMat(self.x[2]).dot(u)
        self.x[2] = _norm_angle(self.x[2])

observationTuple = namedtuple('observationTuple', ['height', 'bearing', 'beacon_id', 'command', 'dt'])

class PolynomialRegression(object):
    def __init__(self, d=3):
        self.d = d
        self.regressor = LinearRegression()
    
    def fit(self, X, y)
        # X is n-dim vector
        # y is n-dim vector
        X_ = np.transpose(np.array([X**i for i in range(1, self.d+1)]))
        y_ = y[:, np.newaxis]
        self.regressor = self.regressor.fit(X_, y_)

    def predict(self, X):
        # X is n-dim vector
        X_ = np.transpose(np.array([X**i for i in range(1, self.d+1)]))
        return self.regressor = self.regressor.predict(X_)

class ActionMapper(object):
    # TODO
    def __init__(self):
        pass

class EM(object):
    def __init__(self):
        self.n_state = 3
        self.n_action = 3
        self.n_meas = 2
        self.cmd_size = cmd_size

        self.forward_model = EKFforward(dim_x=n_state, dim_z=n_meas)
        self.backward_model = BackwardInformationFilter(n_state=n_state, n_meas=n_meas)

        self._create_models()

        self.bpos = {k: v for k, v in enumerate(BEACONS)} 

    def _create_models(self):
        self.sensor_mean_model = PolynomialRegression(d=3)
        self.action_mean_model = ActionMapper(cmd_size, n_action=n_action)
        self.sensor_varn_model = np.diag([10., 0.2])
        self.action_varn_model = np.diag([10., 10., 0.1])

    def _initialize_em(self):
        self.alphas = []
        self.betas = []
        self.gammas = []
        self.deltas = [] # b * beta
        self.forward_model.x = np.array([0., 0., 0.])
        self.forward_model.P = np.diag([1000., 1000., np.pi / 10.]) # belief covariance
        self.forward_model.Q = np.copy(self.action_varn_model)
        self.forward_model.R = np.copy(self.sensor_varn_model)
        self.backward_model.reset()
        self.backward_model.Q = np.copy(self.action_varn_model)
        self.backward_model.R_inv = np.inv(self.sensor_varn_model)

        self.alphas.append((self.forward_model.x, self.forward_model.P))
        self.betas.append((None, np.diag(np.ones((self.n_state, )) * 100000.0)))

    def hx_beacon(self, b):
        # x - current state, b - beacon position
        def hx(x):
            dist = Lnorm(x[:2] - b[:2])
            bear = _norm_angle(math.atan2(b[1]-x[1], b[0]-x[0]) - x[2])
            ot = np.array([self.sensor_model.predict([dist]), bear])
            return ot
        return hx

    def HJacobian_at_beacon(self, b):
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
        self._initialize_em()
        # ==== forward model prediction ====
        for data_t in data:
            ht, bear, bid, cmd, dt = data_t
            act = self.action_model.get(cmd) * dt
            HJacobian_at = self.HJacobian_at_beacon(self.bpos[bid])
            hx = self.hx_beacon(self.bpos[bid])
            if ht is not None:
                obs = np.array([ht, bear])
            else:
                obs = None

            self.forward_model.predict(u=act)
            self.forward_model.update(obs, HJacobian_at, hx)

            self.alphas.append((self.forward_model.x_post, self.forward_model.P_post))

        # ==== backward model prediction ====
        for data_t in reversed(data):
            ht, bear, bid, cmd, dt = data_t
            act = self.action_model.get(cmd) * dt
            HJacobian_at = self.HJacobian_at_beacon(self.bpos[bid])
            hx = self.hx_beacon(self.bpos[bid])
            if ht is not None:
                obs = np.array([ht, bear])
            else:
                obs = None

            self.backward_model.update(obs, HJacobian_at, hx)
            self.deltas.append(self.backward_model.get_params())
            self.backward_model.predict(u=act)
            self.betas.append(self.backward_model.get_params())

        # ==== gamma model prediction ====
        for a, b in zip(self.alphas, self.betas):
            self.gammas.append(mul_gaussians(a, b))

    def Mstep(self, data):
        # data - list of observationTuples
        # Note: height, bearing are after taking command for dt
        # ==== Action model update =====
        mu_cmds = np.zeros((self.cmd_size, self.n_action))
        n_cmds = np.zeros((self.cmd_size, ))
        for i, data_t in enumerate(data_t):
            ht, bear, bid, cmd, dt = data_t
            mu_alpha_t, sigma_alpha_t = self.alphas[i] # TODO - take care of indexing
            mu_delta_t, sigma_delta_t = self.deltas[i] # TODO - take care of indexing
            mu_alpha_delta = np.concatenate((mu_alpha_t, mu_delta_t))
            sigma_alpha_delta = np.zeros((self.n_state*2, self.n_state*2))
            sigma_alpha_delta[0:self.n_state, 0:self.n_state] = sigma_alpha_t
            sigma_alpha_delta[-self.n_state:, -self.n_state:] = sigma_delta_t
            sigma_cmd = self.action_varn_model

            # D(s_t, s_t_1) = L*st_t_1 + m
            # L is Jacobian of D at mu_alpha_delta
            # m = D(mu_alpha_delta) - L*mu_alpha_delta
            P = np.array([[ -1,  0,  0, 1, 0, 0],
                          [  0, -1,  0, 0, 1, 0],
                          [  0,  0, -1, 0, 0, 1]])
            D = lambda s: rotMat(s[2]).dot(np.dot(P, mu_alpha_delta))
            L = np.zeros((self.n_action, self.n_state*2))
            L[:, 0:self.n_state] = -rotMat(mu_alpha_delta[2])
            L[:, self.n_state:2*self.n_state] = rotMat(mu_alpha_delta[2])
            tmp = (lambda t: np.array([[-np.sin(t), -np.cos(t)],
                                       [ np.cos(t), -np.sin(t)]]))(mu_alpha_delta[2])
            L[0:2, 2] = tmp.dot(np.dot(P, mu_alpha_delta)[0:2])
            m = D(mu_alpha_delta) - L.dot(mu_alpha_delta)

            Inv = np.linalg.inv(sigma_cmd + L.dot(sigma_alpha_delta).dot(L.T))
            diff = L.dot(mu_alpha_delta) + m - self.action_model.get(cmd)
            mu_cmds[cmd, :] = self.action_model.get(cmd) + sigma_cmd.dot(Inv).dot(diff)
            n_cmds[cmd] = n_cmds[cmd] + 1

        mu_cmds = mu_cmds / n_cmds.reshape((-1,1)) # is division by zero possible?
        # TODO - update the action_mean_model

        # ==== Sensor model update ==== 
        obs_pred = []
        obs_data = []
        for i, data_t in enumerate(data):
            ht, bear, bid, cmd, dt = data_t
            mu_gamma_t, sigma_gamma_t = self.gammas[i]
            obs_t = hx_beacon(self.bpos[bid])(np.random.multivariate_normal(mu_gamma_t, sigma_gamma_t))
            obs_pred.append(obs_t)
            obs_data.append([ht, bear])
        obs_pred = np.array(obs_pred)
        obs_data = np.array(obs_data)
        self.sensor_mean_model.fit(obs_pred[:, 0], obs_data[:, 0]) # update the regression coefs.

        sigma_1 = np.std(self.sensor_mean_model.predict(obs_pred[:, 0]) - obs_data[:, 0])
        sigma_2 = np.std(obs_pred[:, 1] - obs_data[:, 1])
        self.sensor_varn_model = np.diag([sigma_1**2, sigma_2**2])
        
        

