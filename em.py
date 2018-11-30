from filterpy.kalman import ExtendedKalmanFilter
from numpy.linalg import norm as Lnorm
from collections import namedtuple
from utils import *
from sklearn.linear_model import LinearRegression
from scipy.stats import multivariate_normal

import numpy as np
import pdb
import math
import argparse

FIELD_Y = 3.600#3600
FIELD_X = 5.400#5400

HALF_FIELD_Y = FIELD_Y/2.0
HALF_FIELD_X = FIELD_X/2.0

BEACONS = [(HALF_FIELD_X, HALF_FIELD_Y),       #  WO_BEACON_BLUE_YELLOW
           (HALF_FIELD_X, -HALF_FIELD_Y),      #  WO_BEACON_YELLOW_BLUE,
           (0, HALF_FIELD_Y),                  #  WO_BEACON_BLUE_PINK
           (0, -HALF_FIELD_Y),                 #  WO_BEACON_PINK_BLUE
           (-HALF_FIELD_X, HALF_FIELD_Y),      #  WO_BEACON_PINK_YELLOW
           (-HALF_FIELD_X, -HALF_FIELD_Y)]     #  WO_BEACON_YELLOW_PINK,

def _norm_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))

class EKFforward(ExtendedKalmanFilter):
    def __init__(self, *args, action_cov=None, **kwargs):
        super().__init__(*args, **kwargs)
        if action_cov is None:
            self.action_cov = np.eye(self.dim_u)
        else:
            self.action_cov = action_cov

    def predict_x(self, u=0):
        j13 = -np.sin(self.x[2])*u[0] - np.cos(self.x[2])*u[1]
        j23 =  np.cos(self.x[2])*u[0] - np.sin(self.x[2])*u[1]

        self.F = np.array([[1., 0., j13],
                           [0., 1., j23],
                           [0., 0., 1. ]])
        R = rotMat(self.x[2])
        self.Q = R.dot(self.action_cov).dot(R.T)
        # print('X before predict: {}'.format(self.x))
        self.x = self.x + R.dot(u)
        # print('X after predict: {}'.format(self.x))
        self.x[2] = _norm_angle(self.x[2])

    def get_params(self):
        return (np.copy(self.x), np.copy(self.P))

class EKFbackward(ExtendedKalmanFilter):
    def __init__(self, *args, action_cov=None, **kwargs):
        super().__init__(*args, **kwargs)
        if action_cov is None:
            self.action_cov = np.eye(self.dim_u)
        else:
            self.action_cov = action_cov

    def predict_x(self, u=0):
        th = self.x[2] - u[2]
        j13 = -np.sin(th)*u[0] - np.cos(th)*u[1]
        j23 =  np.cos(th)*u[0] - np.sin(th)*u[1]

        self.F = np.array([[1., 0., j13],
                           [0., 1., j23],
                           [0., 0., 1. ]])
        R = rotMat(self.x[2] - u[2])
        self.Q = R.dot(self.action_cov).dot(R.T)
        # print('X before predict: {}'.format(self.x))
        self.x = self.x - R.dot(u)
        # print('X after predict: {}'.format(self.x))
        self.x[2] = _norm_angle(self.x[2])

    def get_params(self):
        return (np.copy(self.x), np.copy(self.P))

class PolynomialRegression(object):
    def __init__(self, d=3):
        self.d = d
        self.regressor = LinearRegression()
        # self.fit(np.linspace(2000,4000, 1000), np.linspace(40,0, 1000))
        self.fit(np.linspace(2.000,4.000, 1000), np.linspace(40,0, 1000))

    def fit(self, X, y):
        # X is n-dim vector
        # y is n-dim vector
        X_ = np.zeros((len(X), self.d))
        for i in range(1, self.d+1):
            X_[:, i-1] = np.array(X)**i
        y_ = np.array(y)[:, np.newaxis]
        self.regressor = self.regressor.fit(X_, y_)

    def predict(self, X):
        # X is n-dim vector
        X_ = np.zeros((len(X), self.d))
        for i in range(1, self.d+1):
            X_[:, i-1] = np.array(X)**i
        return self.regressor.predict(X_).squeeze(axis=(1,))

class ActionMapper(object):
    def __init__(self, cmd_size=40, n_action=3):
        self.cmd_size = 40
        self.n_action = n_action
        self.cmd_mus = np.zeros((cmd_size, n_action))

    def get(self, cmd):
        return self.cmd_mus[cmd]

    def update(self, new_mus):
        self.cmd_mus = np.copy(new_mus)
        return

class EM(object):
    def __init__(self):
        self.n_state = 3
        self.n_action = 3
        self.n_meas = 2
        self.cmd_size = 40

        self._create_models()
        self.forward_model = EKFforward(action_cov=self.action_varn_model.copy(),
                                        dim_x=self.n_state, dim_z=self.n_meas)
        self.backward_model = EKFbackward(action_cov=self.action_varn_model.copy(),
                                            dim_x=self.n_state, dim_z=self.n_meas)

        self.bpos = {k: np.array(v) for k, v in enumerate(BEACONS)}

    def _create_models(self):
        self.sensor_mean_model = PolynomialRegression(d=3)
        self.action_mean_model = ActionMapper(cmd_size=self.cmd_size, n_action=self.n_action)
        self.sensor_varn_model = np.diag([100., 0.04])
        # self.action_varn_model = np.diag([100., 100., 0.01])
        # self.prior_mean_model  = np.zeros((self.n_state,))
        # self.prior_varn_model  = np.diag([10000., 10000., np.pi / 10.])
        self.action_varn_model = np.diag([1e-4, 1e-4, 0.01])
        self.prior_mean_model  = np.zeros((self.n_state,))
        self.prior_varn_model  = np.diag([0.01, 0.01, np.pi / 10.])

    def _initialize_em(self):
        self.alphas = []
        self.betas = []
        self.gammas = []
        self.deltas = [] # b * beta
        self.forward_model.x = np.array([0., 0., 0.])
        # self.forward_model.P = np.diag([10000., 10000., np.pi / 10.]) # belief covariance
        self.forward_model.P = np.diag([0.01, 0.01, np.pi / 10.]) # belief covariance
        self.forward_model.Q = np.copy(self.action_varn_model)
        self.forward_model.R = np.copy(self.sensor_varn_model)
        self.backward_model.x = np.array([0., 0., 0.])
        # self.backward_model.P = np.diag([10000000., 10000000., 100*np.pi]) # belief covariance
        self.backward_model.P = np.diag([10, 10., 100*np.pi]) # belief covariance
        self.backward_model.Q = np.copy(self.action_varn_model)
        self.backward_model.R = np.copy(self.sensor_varn_model)

        self.alphas.append((self.prior_mean_model.copy(), self.prior_varn_model.copy()))
        self.betas.append((self.backward_model.x, self.backward_model.P))
        self.log_likelihood = 0.0

    def hx_beacon(self, b):
        # x - current state, b - beacon position
        def hx(x):
            dist = Lnorm(x[:2] - b[:2])
            bear = _norm_angle(math.atan2(b[1]-x[1], b[0]-x[0]) - x[2])
            ot = np.array([self.sensor_mean_model.predict([dist])[0], bear])
            return ot
        return hx

    def HJacobian_at_beacon(self, b):
        # x - current state, b - beacon position
        def HJacobian_at(x):
            dx = x[:2] - b[:2]
            dist = Lnorm(dx)
            sensor_params = self.sensor_mean_model.regressor.coef_[0].copy()
            dim = self.sensor_mean_model.d
            dhx = (sensor_params * np.arange(1, dim+1)).dot(np.array([dist**i for i in range(0, dim)]))
            return np.array([[dhx*dx[0]/dist, dhx*dx[1]/dist,  0],
                             [-dx[1]/dist**2,  dx[0]/dist**2, -1]])
        return HJacobian_at

    def Estep(self, data):
        # data - list of observationTuples
        # Note: height, bearing are after taking command for dt
        self._initialize_em()
        # ==== forward model prediction ====
        for i, data_t in enumerate(data):
            print('=====> Forward {}/{}'.format(i+1, len(data)))
            ht, bear, bid, cmd, dt = data_t
            act = self.action_mean_model.get(cmd) * dt
            # print('idx : {}, ht: {}, bear: {}, bid: {}, act: {}'.format(i, ht, bear, bid, act))
            # print('X initial: {}, obs: {}'.format(self.forward_model.x, obs))
            self.forward_model.predict(u=act)
            # self.forward_model.x = self.forward_model.x.clip([-FIELD_X*2, -FIELD_Y*2, -4], [FIELD_X*2, FIELD_Y*2, 4])

            HJacobian_at = None
            hx = None
            if ht is not None:
                obs = np.array([ht, bear])
                HJacobian_at = self.HJacobian_at_beacon(self.bpos[bid])
                hx = self.hx_beacon(self.bpos[bid])

                # log likelihood computation
                mu_alpha_, sigma_alpha_ = self.forward_model.get_params()
                H = HJacobian_at(mu_alpha_)
                mu_obs_t = H.dot(mu_alpha_)
                sigma_obs_t = H.dot(sigma_alpha_).dot(H.T) + self.sensor_varn_model
                self.log_likelihood += multivariate_normal.logpdf(obs, mean=mu_obs_t, cov=sigma_obs_t)
            else:
                obs = None

            if obs is not None:
                print('====> Estep: bid: {}, bpos[bid]: {}'.format(bid, self.bpos[bid]))
                print('====> Estep: sensor params: {}, {}'.format(self.sensor_mean_model.regressor.coef_, self.sensor_mean_model.regressor.intercept_))
            self.forward_model.update(obs, HJacobian_at, hx)
            # self.forward_model.x = self.forward_model.x.clip([-FIELD_X*2, -FIELD_Y*2, -4], [FIELD_X*2, FIELD_Y*2, 4])

            self.alphas.append(self.forward_model.get_params())

        # ==== backward model prediction ====
        for i, data_t in enumerate(list(reversed(data))):
            print('=====> Backward {}/{}'.format(i+1, len(data)))
            ht, bear, bid, cmd, dt = data_t
            act = self.action_mean_model.get(cmd) * dt
            HJacobian_at = None
            hx = None
            if ht is not None:
                obs = np.array([ht, bear])
                HJacobian_at = self.HJacobian_at_beacon(self.bpos[bid])
                hx = self.hx_beacon(self.bpos[bid])
            else:
                obs = None

            self.backward_model.update(obs, HJacobian_at, hx)
            # self.backward_model.x = self.backward_model.x.clip([-FIELD_X*2, -FIELD_Y*2, -4], [FIELD_X*2, FIELD_Y*2, 4])
            self.deltas.append(self.backward_model.get_params())
            self.backward_model.predict(u=act)
            # self.backward_model.x = self.backward_model.x.clip([-FIELD_X*2, -FIELD_Y*2, -4], [FIELD_X*2, FIELD_Y*2, 4])
            self.betas.append(self.backward_model.get_params())

        # ==== gamma model prediction ====
        for a, b in zip(self.alphas, self.betas):
            gamma = mul_gaussians(a, b)
            gamma[0][2] = _norm_angle(gamma[0][2])
            self.gammas.append(gamma)

    def Mstep(self, data):
        # data - list of observationTuples
        # Note: height, bearing are after taking command for dt
        # ==== Action model update =====
        mu_cmds = np.zeros((self.cmd_size, self.n_action))
        n_cmds = np.zeros((self.cmd_size, ))
        for i, data_t in enumerate(data): # Note: t = i+1
            ht, bear, bid, cmd, dt = data_t
            mu_alpha_t_1, sigma_alpha_t_1 = self.alphas[i]
            mu_delta_t, sigma_delta_t = self.deltas[i]
            mu_alpha_delta = np.concatenate((mu_alpha_t_1, mu_delta_t))
            sigma_alpha_delta = np.zeros((self.n_state*2, self.n_state*2))
            sigma_alpha_delta[0:self.n_state, 0:self.n_state] = sigma_alpha_t_1
            sigma_alpha_delta[-self.n_state:, -self.n_state:] = sigma_delta_t
            sigma_cmd = self.action_varn_model

            # D(s_t, s_t_1) = L*st_t_1 + m
            # L is Jacobian of D at mu_alpha_delta
            # m = D(mu_alpha_delta) - L*mu_alpha_delta
            P = np.array([[ -1,  0,  0, 1, 0, 0],
                          [  0, -1,  0, 0, 1, 0],
                          [  0,  0, -1, 0, 0, 1]])
            D = lambda s: rotMat(-s[2]).dot(np.dot(P, s))
            L = np.zeros((self.n_action, self.n_state*2))
            L[:, 0:self.n_state] = -rotMat(mu_alpha_delta[2])
            L[:, self.n_state:2*self.n_state] = rotMat(mu_alpha_delta[2])
            tmp = (lambda t: np.array([[-np.sin(t), -np.cos(t)],
                                       [ np.cos(t), -np.sin(t)]]))(mu_alpha_delta[2])
            L[0:2, 2] = tmp.dot(np.dot(P, mu_alpha_delta)[0:2])
            m = D(mu_alpha_delta) - L.dot(mu_alpha_delta)

            print('====> Mstep progress: {}/{}'.format(i, len(data)))
            print('====> Mstep:\nmu_alpha_delta: {}\nsigma_alpha_delta: {}\nL: {}'.format(mu_alpha_delta, sigma_alpha_delta, L))
            Inv = np.linalg.inv(sigma_cmd + L.dot(sigma_alpha_delta).dot(L.T))
            diff = L.dot(mu_alpha_delta) + m - self.action_mean_model.get(cmd)
            mu_cmds[cmd, :] = self.action_mean_model.get(cmd) + sigma_cmd.dot(Inv).dot(diff)
            n_cmds[cmd] = n_cmds[cmd] + 1

        mu_cmds = mu_cmds / (n_cmds.reshape((-1,1)) + 1e-8)
        mu_cmds[n_cmds == 0] = 0

        self.action_mean_model.update(mu_cmds)

        # ==== Sensor model update ==== 
        regressX = []
        regressY = []
        for i, data_t in enumerate(data):
            ht, bear, bid, cmd, dt = data_t
            mu_gamma_t, sigma_gamma_t = self.gammas[i+1]
            if ht is not None:
                s_t = np.random.multivariate_normal(mu_gamma_t, sigma_gamma_t)
                dist_t = Lnorm(s_t[:2] - self.bpos[bid][:2])
                ang_t = self.hx_beacon(self.bpos[bid])(s_t)[1]
                regressX.append([dist_t, ang_t])
                regressY.append([ht, bear])
        regressX = np.array(regressX)
        regressY = np.array(regressY)
        self.sensor_mean_model.fit(regressX[:, 0], regressY[:, 0]) # update the regression coefs.

        sigma_1 = np.std(self.sensor_mean_model.predict(regressX[:, 0]) - regressY[:, 0])
        sigma_2 = np.std(regressX[:, 1] - regressY[:, 1])
        self.sensor_varn_model = np.diag([sigma_1**2, sigma_2**2])
        self.prior_mean_model  = np.copy(self.gammas[0][0])
        self.prior_varn_model  = np.copy(self.gammas[0][1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='2d_asami_data.txt')
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    np.random.seed(args.seed)

    em = EM()
    data = preprocess_data(args.data)

    for it in range(0, args.n_iter):
        print('=====> Iteration {:d}'.format(it))
        em.Estep(data)
        print('=====> Log Likelihood {:.3f}'.format(em.log_likelihood))
        em.Mstep(data)
