import numpy as np
from utils import rotMat, _norm_angle

class BackwardInformationFilter(object):
    def __init__(self, n_state=3, n_meas=2):
        self.n_state = n_state
        self.n_meas = n_meas
        self.R_inv = np.zeros((n_meas, n_meas))   # sensor noise
        self.Q = np.diag(np.ones((n_state,)))     # dynamics noise
        self.reset()

    def update(self, z, HJacobian_at, hx):
        if z is not None:
            self.P_inv = self.P_inv + np.dot(HJacobian_at.T, self.R_inv).dot(HJacobian_at)

            if self.seen_obs:
                z_err = z - hx(self.mu) + HJacobian_at.dot(self.mu)
            else:
                z_err = z
                self.seen_obs = True

            self.eps = self.eps + np.dot(HJacobian_at.T.dot(self.R_inv), z_err)


    def predict(self, u):
        # Note: u must be (delta_x, delta_y and delta_theta) from the forward step
        if self.seen_obs:
            self.mu = np.linalg.inv(self.P_inv).dot(self.eps)
            self.G = self.computeG(self.mu, u)
            self.P_inv = np.linalg.inv(self.G.dot(self.P_inv).dot(self.G.T) + self.Q)
            self.mu = self.g(self.mu, u)
            self.eps = self.P_inv.dot(self.mu)
        else:
            self.eps = np.zeros((self.n_state, ))
            self.P_inv = np.zeros((self.n_state, self.n_state))

    def g(self, mu, u):
        # Inverse dynamics model
        mu = mu - u * np.array([0., 0., 1.]) - rotMat(mu[2]-u[2]).dot(u * np.array([1., 1., 0.]))
        mu[2] = _norm_angle(mu[2])
        return mu

    def computeG(self, mu, u):
        th = mu[2] - u[2]
        j13 = -np.sin(th)*u[0] + np.cos(th)*u[1]
        j23 = -np.cos(th)*u[0] - np.sin(th)*u[1]
        G = np.array([[1., 0., j13], 
                      [0., 1., j23],
                      [0., 0., 1. ]])
        return G

    def reset(self):
        self.G = np.zeros((self.n_state, self.n_state))     # dynamics jacobian
        self.eps = np.zeros((self.n_state,))           # information vector
        self.P_inv = np.zeros((self.n_state, self.n_state)) # inverse belief covariance / information matrix
        self.mu = None
        self.seen_obs = False                     # was the first observation seen

    def get_params(self):
        if self.seen_obs:
            P = np.linalg.inv(self.P_inv)
            mu = P.dot(self.eps)
        return mu, P
