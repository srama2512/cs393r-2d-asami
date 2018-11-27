import numpy as np

class InformationFilter(object):
    def __init__(self, n_state=3, n_meas=2):
        self.G = np.zeros((n_state, n_state))  # dynamics jacobian
        self.R_inv = np.zeros((n_meas, n_meas))    # sensor noise
        self.Q = np.diag(np.ones((n_state,)))  # dynamics noise
        self.eps = np.zeros((n_state,))        # information vector
        self.P_inv = np.zeros((n_state, n_state)) # inverse belief covariance / information matrix

    def update(self, z, HJacobian_at, hx):
        self.P_inv = self.P_inv + np.dot(HJacobian_at.T, self.R_inv).dot(HJacobian_at)
        z_err = z - hx(self.mu) - HJacobian_at.dot(self.mu)
        self.eps = self.eps + np.dot(HJacobian_at.T.dot(self.R_inv), z_err)

    def predict(self, u):
        self.mu = np.linalg.inv(self.P_inv).dot(self.eps)
        self.P_inv = np.linalg.inv(self.G.dot(self.P_inv).dot(self.G.T) + self.Q)
        # TODO dyn
        self.mu = self.dyn(u, self.mu)
        self.eps = self.P_inv.dot(self.mu)
