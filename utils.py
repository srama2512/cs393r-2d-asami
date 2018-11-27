import numpy as np

def rotMat(theta):
    return np.array([[np.cos(theta), -np.sin(theta),         0.],
                     [np.sin(theta),  np.cos(theta),         0.],
                     [           0.,             0.,         1.]])
