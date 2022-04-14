import numpy as np

def create_mesh(n, D=0):
    L, W = 7, 7
    kx, ky = 31, 31
 
    n = n / np.linalg.norm(n)
    x, y = np.mgrid[-L:L:kx*1j, -W:W:ky*1j]
    A, B, C = n
    z = -(A*x+B*y+D)/C
    return x, y, z

def create_point_generator(vec_par, vec_offset, D):
    def generator(n, t):
        return vec_offset * n + vec_par * t + D
    return generator