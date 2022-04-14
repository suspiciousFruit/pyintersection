import numpy as np
import pyintersection as pyi
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from interpolations import interpolate_rpoints

def plot_rpoints(axis, rpoints, c=None, s=None, **kwargs):
    ix, iy, iz = rpoints[:, 4:5], rpoints[:, 5:6], rpoints[:, 6:7]
    axis.scatter(ix, iy, iz, c=c, s=s, **kwargs)
    
def plot_mpoints(axis, mpoints, c=None, s=None, **kwargs):
    ix, iy, iz = mpoints[:, 2:3], mpoints[:, 3:4], mpoints[:, 4:5]
    axis.scatter(ix, iy, iz, c=c, s=s, **kwargs)
    
def plot_rawpoints(axis, points, c=None, s=None, **kwargs):
    ix, iy, iz = points[:, 0:1], points[:, 1:2], points[:, 2:3]
    axis.scatter(ix, iy, iz, c=c, s=s, **kwargs)
    
def plot_surface(axis, mesh, **kwargs):
    ix, iy, iz = mesh
    axis.plot_surface(ix, iy, iz, **kwargs)

def create_sphere_mesh():
    r = 5.0
    L, W = 2*np.pi, np.pi
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)*r
    y = np.sin(u)*np.sin(v)*r
    z = np.cos(v)*r
    return x, y, z

def create_plain_mesh(n, D=0):
    L, W = 7, 7
    kx, ky = 31, 31
 
    n = n / np.linalg.norm(n)
    x, y = np.mgrid[-L:L:kx*1j, -W:W:ky*1j]
    A, B, C = n
    z = -(A*x+B*y+D)/C
    return (x, y, z)

def create_manifold_generator(generate_point=None, points_count=40):
    scaling = 1
    def generator(n, t, atol, points_count=points_count):
        size = int(points_count**0.5)
        twidth = atol * scaling
        nwidth = atol * scaling
        ts = np.linspace(t - twidth / 2, t + twidth / 2, size)
        ns = np.linspace(n - nwidth / 2, n + nwidth / 2, size)
        res = np.zeros((len(ns) * len(ts), 5))

        for i, n in enumerate(ns):
            for j, t in enumerate(ts):
                res[j + len(ts) * i] = n, t, *generate_point(n, t)
        return res
    return generator

def create_point_generator(vec_par, vec_offset):
    def generator(n, t):
        return vec_offset * n + vec_par * t
    return generator

def sphere_point_generator(n, t):
    r = 5.0
    return np.array([
        np.cos(n)*np.sin(t)*r,
        np.sin(n)*np.sin(t)*r,
        np.cos(t)*r
    ])



plain_norm = np.array([1, 1, 1]) # Norm vector of the plane
plain_par = np.array([1, -0.5, -0.5]) # Some orthogonal to norm vector
plain_offset = np.array([0, 1, -1]) # One another orthogonal to previous other vectors
plain_point_generator = create_point_generator(plain_par, plain_offset)
plain_manifold_generator = create_manifold_generator(plain_point_generator, points_count=80)
plain_mesh = create_plain_mesh(plain_norm)
sphere_manifold_generator = create_manifold_generator(sphere_point_generator, points_count=80)
sphere_mesh = create_sphere_mesh()



plain = plain_manifold_generator(0, 0, 10)
sphere = sphere_manifold_generator(0, 0, 3)
_, rpoints, _ = pyi.optimize3d(plain, sphere, 0.1,
    plain_manifold_generator, sphere_manifold_generator, max_iter=3)


_, nts = interpolate_rpoints(rpoints, 0)
points = []
for i in range(len(nts[0])):
    points.append(plain_point_generator(nts[0][i], nts[1][i]))
points = np.array(points)


# Plot results
fig = plt.figure(figsize=(12, 12))
axis = fig.add_subplot(111, projection='3d')
plot_surface(axis, plain_mesh, alpha=0.5)
plot_surface(axis, sphere_mesh, alpha=0.5)
plot_rawpoints(axis, points, c='g', s=4)
plt.show()
