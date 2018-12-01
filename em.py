from filterpy.kalman import ExtendedKalmanFilter
from numpy.linalg import norm as Lnorm
from collections import namedtuple
from utils import *
from sklearn.linear_model import LinearRegression
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pdb
import math
import argparse

FIELD_Y = 3600#3600
FIELD_X = 5400#5400

gt_data = open('state_log.txt', 'r').read().split('\n')[1:-1]
gt_data = np.array([[float(k) for k in d.split(', ')] for d in gt_data])

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

def _norm_angle_vec(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

def residual_fn(z, z2):
    dz = z - z2
    dz[1] = math.atan2(math.sin(dz[1]), math.cos(dz[1]))
    return dz

class EKFforward(ExtendedKalmanFilter):
    def __init__(self, *args, action_cov=None, **kwargs):
        super().__init__(*args, **kwargs)
        if action_cov is None:
            self.action_cov = np.eye(self.dim_u)
        else:
            self.action_cov = action_cov

    def predict_x(self, u=0):
        j13 = -np.sin(self.x[2])*u[0] + np.cos(self.x[2])*u[1]
        j23 = -np.cos(self.x[2])*u[0] - np.sin(self.x[2])*u[1]

        self.F = np.array([[1., 0., j13],
                           [0., 1., j23],
                           [0., 0., 1. ]])
        R = rotMat(-self.x[2])
        self.Q = R.dot(self.action_cov).dot(R.T)
        self.x = self.x + R.dot(u)
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
        j13 =  np.sin(th)*u[0] - np.cos(th)*u[1]
        j23 =  np.cos(th)*u[0] + np.sin(th)*u[1]

        self.F = np.array([[1., 0., j13],
                           [0., 1., j23],
                           [0., 0.,  1.]])
        R = rotMat(-th)
        self.Q = R.dot(self.action_cov).dot(R.T)
        self.x = self.x - R.dot(u)
        self.x[2] = _norm_angle(self.x[2])

    def get_params(self):
        return (np.copy(self.x), np.copy(self.P))

class PolynomialRegression(object):
    def __init__(self, d=3):
        self.d = d
        self.regressor = LinearRegression()
        d = np.linspace(1500, 4000, 1000)
        ht = 10*np.exp(2.-d/2000.)
        #ht = 5*np.exp(2.-d/2000.)
        #ht = np.linspace(1500, 4000, 1000)

        self.fit(d, ht)

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
        self.b_angles = [0, math.pi, math.pi/4, -math.pi/4, math.pi/2, -math.pi/2, 3*math.pi/4, -3*math.pi/4]
        self.a_vels = [-1./2., -1./6., 0., 1./6., 1./2.]
        self.gt_mus = np.zeros((cmd_size, n_action))
        count = 0
        for a in self.a_vels:
            for b in self.b_angles:
                magn = math.sqrt(1-a**2)
                vx = magn*math.cos(b)
                vy = magn*math.sin(b)
                self.gt_mus[count] = np.array(list(self.getGTVelocities(vx, vy, a)))
                count += 1

        self.cmd_mus = self.gt_mus.copy()

    def getGTVelocities(self, x, y, theta):
        return x*240.0, y*120.0, theta*math.radians(130.0)

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
        self.action_varn_model = np.diag([4000., 4000., 0.25])
        self.prior_mean_model  = np.zeros((self.n_state,))
        self.prior_varn_model  = np.diag([10000., 10000., np.pi / 10.])

    def _initialize_em(self):
        self.alphas = []
        self.betas = []
        self.gammas = []
        self.deltas = [] # b * beta
        self.forward_model.x = np.copy(self.prior_mean_model)
        self.forward_model.P = np.copy(self.prior_varn_model) # belief covariance
        self.forward_model.Q = np.copy(self.action_varn_model)
        self.forward_model.R = np.copy(self.sensor_varn_model)
        self.backward_model.x = np.array([0., 0., 0.])
        self.backward_model.P = np.diag([100000., 100000., 2*np.pi]) # belief covariance
        self.backward_model.Q = np.copy(self.action_varn_model)
        self.backward_model.R = np.copy(self.sensor_varn_model)

        self.alphas.append((self.prior_mean_model.copy(), self.prior_varn_model.copy()))
        self.betas.append((self.backward_model.x, self.backward_model.P))
        self.log_likelihood = 0.0

    def hx_beacon(self, b):
        # x - current state, b - beacon position
        def hx(x):
            dist = Lnorm(x[:2] - b[:2])
            rel_pos_beacon = (rotMat(x[2])[:2, :2]).dot(b[:2] - x[:2])
            bear = math.atan2(rel_pos_beacon[1], rel_pos_beacon[0])
            ot = np.array([self.sensor_mean_model.predict([dist])[0], bear])
            return ot
        return hx

    def HJacobian_at_beacon(self, b): # df(x)/dx
        # x - current state, b - beacon position
        def HJacobian_at(x):
            dx = x[:2] - b[:2]
            dist = Lnorm(dx)
            sensor_params = self.sensor_mean_model.regressor.coef_[0].copy()
            dim = self.sensor_mean_model.d
            dhx = (sensor_params * np.arange(1, dim+1)).dot(np.array([dist**i for i in range(0, dim)]))
            rel_pos_beacon = (rotMat(x[2])[:2, :2]).dot(b[:2] - x[:2])
            rpx, rpy = rel_pos_beacon
            yx2y2 = rpy/(rpx**2 + rpy**2)
            xx2y2 = rpx/(rpx**2 + rpy**2)
            j21 = np.dot([yx2y2, xx2y2], [ np.cos(x[2]), -np.sin(x[2])])
            j22 = np.dot([yx2y2, xx2y2], [-np.sin(x[2]), -np.cos(x[2])])
            return np.array([[dhx*dx[0]/dist, dhx*dx[1]/dist,  0],
                             [           j21,            j22,  1]])
        return HJacobian_at

    def Estep(self, data):
        # data - list of observationTuples
        # Note: height, bearing are after taking command for dt
        self._initialize_em()
        self.forward_distance_estimates = []
        self.forward_observations = []
        self.backward_distance_estimates = []
        self.backward_observations = []
        # ==== forward model prediction ====
        for i, data_t in enumerate(data):
            ht, bear, bid, cmd, dt, _ = data_t
            act = self.action_mean_model.get(cmd) * dt
            self.forward_model.predict(u=act)

            HJacobian_at = None
            hx = None
            if ht is not None:
                obs = np.array([ht, bear])
                HJacobian_at = self.HJacobian_at_beacon(self.bpos[bid])
                hx = self.hx_beacon(self.bpos[bid])
            else:
                obs = None

            self.forward_model.update(obs, HJacobian_at, hx, residual=residual_fn)

            if ht is not None:
                # log likelihood computation
                mu_alpha, sigma_alpha = self.forward_model.get_params()
                H = HJacobian_at(mu_alpha)
                mu_obs_t = H.dot(mu_alpha) + hx(mu_alpha) - H.dot(mu_alpha)
                sigma_obs_t = H.dot(sigma_alpha).dot(H.T) + self.sensor_varn_model
                self.log_likelihood += multivariate_normal.logpdf(obs, mean=mu_obs_t, cov=sigma_obs_t)

            forward_params = self.forward_model.get_params()
            self.alphas.append(forward_params)
            if obs is not None:
                d_to_beacon = Lnorm(forward_params[0][:2] - self.bpos[bid][:2])
                self.forward_distance_estimates.append(d_to_beacon)
                self.forward_observations.append(d_to_beacon)

        # ==== backward model prediction ====
        for i, data_t in enumerate(list(reversed(data))):
            ht, bear, bid, cmd, dt, _ = data_t
            act = self.action_mean_model.get(cmd) * dt
            HJacobian_at = None
            hx = None
            if ht is not None:
                obs = np.array([ht, bear])
                HJacobian_at = self.HJacobian_at_beacon(self.bpos[bid])
                hx = self.hx_beacon(self.bpos[bid])
            else:
                obs = None

            self.backward_model.update(obs, HJacobian_at, hx, residual=residual_fn)
            self.deltas.append(self.backward_model.get_params())
            self.backward_model.predict(u=act)
            backward_params = self.backward_model.get_params()
            self.betas.append(backward_params)
            if obs is not None:
                d_to_beacon = Lnorm(backward_params[0][:2] - self.bpos[bid][:2])
                self.backward_distance_estimates.append(d_to_beacon)
                self.backward_observations.append(d_to_beacon)

        self.deltas = list(reversed(self.deltas))
        self.betas  = list(reversed(self.betas))
        self.backward_distance_estimates = list(reversed(self.backward_distance_estimates))
        self.backward_observations = list(reversed(self.backward_observations))

        # ==== gamma model prediction ====
        for a, b in zip(self.alphas, self.betas):
            gamma = deepcopy(a)#mul_gaussians(a, b)
            gamma[0][2] = _norm_angle(gamma[0][2])
            self.gammas.append(gamma)

    def Mstep(self, data):
        # data - list of observationTuples
        # Note: height, bearing are after taking command for dt
        # ==== Action model update =====
        mu_cmds = np.zeros((self.cmd_size, self.n_action + 1)) # averaging cosines and sines
        n_cmds = np.zeros((self.cmd_size, ))
        for i, data_t in enumerate(data): # Note: t = i+1
            ht, bear, bid, cmd, dt, _ = data_t
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
            D = lambda s: rotMat(s[2]).dot(np.dot(P, s))
            L = np.zeros((self.n_action, self.n_state*2))
            L[:, 0:self.n_state] = -rotMat(mu_alpha_delta[2])
            L[:, self.n_state:2*self.n_state] = rotMat(mu_alpha_delta[2])
            tmp = (lambda t: np.array([[-np.sin(t), -np.cos(t)],
                                       [ np.cos(t), -np.sin(t)]]))(mu_alpha_delta[2])
            L[0:2, 2] = tmp.dot(np.dot(P, mu_alpha_delta)[0:2])
            m = D(mu_alpha_delta) - L.dot(mu_alpha_delta)

            Inv = np.linalg.inv(sigma_cmd + L.dot(sigma_alpha_delta).dot(L.T))
            diff = L.dot(mu_alpha_delta) + m - self.action_mean_model.get(cmd)
            diff[2] = _norm_angle(diff[2])
            new_cmd = self.action_mean_model.get(cmd) + sigma_cmd.dot(Inv).dot(diff)
            mu_cmds[cmd, :2] += new_cmd[:2]
            mu_cmds[cmd,  2] += np.cos(new_cmd[2])
            mu_cmds[cmd,  3] += np.sin(new_cmd[2])
            n_cmds[cmd] = n_cmds[cmd] + 1

        mu_cmds = mu_cmds / (n_cmds.reshape((-1,1)) + 1e-8)
        mu_cmds[n_cmds == 0] = 0

        mu_cmds = np.concatenate([mu_cmds[:, :2], np.arctan2(mu_cmds[:, 3], mu_cmds[:, 2])[:, np.newaxis]], axis=1)

        self.action_mean_model.update(mu_cmds)

        # ==== Sensor model update ==== 
        regressX = []
        regressY = []
        for i, data_t in enumerate(data):
            ht, bear, bid, cmd, dt, _ = data_t
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
        sigma_1_sqr = np.sum((self.sensor_mean_model.predict(regressX[:, 0]) - regressY[:, 0])**2)
        sigma_2_sqr = np.sum(_norm_angle_vec(regressX[:, 1] - regressY[:, 1])**2)
        self.sensor_varn_model = np.diag([sigma_1_sqr, sigma_2_sqr])
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
        #plt.subplot(2, 1, 1)
        #plt.plot(em.forward_distance_estimates, em.forward_observations)
        #plt.title('Forward pass estimates')
        #plt.subplot(2, 1, 2)
        #plt.plot(em.backward_distance_estimates, em.backward_observations)
        #plt.title('Backward pass estimates')
        #plt.show()

        if (it+1) % 15 == 0:
            fig = plt.figure(0)
            ax = fig.add_subplot(121, projection='3d')
            x = [alpha[0][0] for alpha in em.alphas]
            y = [alpha[0][1] for alpha in em.alphas]
            t = list(range(len(em.alphas)))
            ax.plot(x, y, t, label='alphas')
            x = gt_data[:, 0]
            y = gt_data[:, 1]
            t = range(gt_data.shape[0])
            ax.plot(x, y, t, label='gt curve', linestyle='dashed')
            ax = fig.add_subplot(122, projection='3d')
            x = [beta[0][0] for beta in em.betas]
            y = [beta[0][1] for beta in em.betas]
            t = list(range(len(em.betas)))
            ax.plot(x, y, t, label='betas')
            x = gt_data[:, 0]
            y = gt_data[:, 1]
            t = range(gt_data.shape[0])
            ax.plot(x, y, t, label='gt curve', linestyle='dashed')
            plt.legend()

            fig = plt.figure(1)
            plt.subplot(2, 1, 1)
            theta = [alpha[0][2] for alpha in em.alphas]
            t = list(range(len(em.alphas)))
            plt.plot(t, theta, label='alphas')
            plt.plot(range(gt_data.shape[0]), gt_data[:, 2], linestyle='dashed')
            plt.title('Alphas theta value')
            plt.legend()
            plt.subplot(2, 1, 2)
            theta = [beta[0][2] for beta in em.betas]
            t = list(range(len(em.betas)))
            plt.plot(t, theta, label='betas')
            plt.title('Betas theta value')
            plt.plot(range(gt_data.shape[0]), gt_data[:, 2], linestyle='dashed')
            plt.legend()

            fig = plt.figure(2)
            distances = np.linspace(1000, 4000, 1000)
            plt.plot(distances, em.sensor_mean_model.predict(distances))
            gt_data_pairs = filter(lambda x: x[0] is not None, [[d.height, d.dist] for d in data])
            gt_data_pairs = np.array(list(gt_data_pairs))
            gt_regress = PolynomialRegression(d=3)
            gt_regress.fit(gt_data_pairs[:, 1], gt_data_pairs[:, 0])
            gt_fit = gt_regress.predict(distances)
            plt.plot(distances, gt_fit, linestyle='dashed')
            plt.xlabel('Distances')
            plt.ylabel('Observations')
            plt.title('Learned sensor model plot')
            plt.show()

        em.Mstep(data)
