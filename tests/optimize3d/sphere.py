import numpy as np

def create_mesh():
    r = 5.0
    L, W = 2*np.pi, np.pi
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)*r
    y = np.sin(u)*np.sin(v)*r
    z = np.cos(v)*r
    return x, y, z

def create_point_generator(r=5.0):
    def generator(n, t):
        return np.array([
            np.cos(n)*np.sin(t)*r,
            np.sin(n)*np.sin(t)*r,
            np.cos(t)*r
        ])
    return generator