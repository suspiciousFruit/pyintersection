######################################################################################
#
# Visual test for pyintersection.optimize3d() method.
# The intersection of two planes is being clarified.
#
#######################################################################################

import numpy as np
import pyintersection as pyi
import matplotlib.pyplot as plt

def create_plain(n, D=0):
    L, W = 4, 4
    kx, ky = 31, 31
 
    n = n / np.linalg.norm(n)
    x, y = np.mgrid[-L:L:kx*1j, -W:W:ky*1j]
    A, B, C = n
    z = -(A*x+B*y+D)/C
    return (x, y, z)

def plot_rpoints(axis, rpoints, c=None, s=None):
    ix, iy, iz = rpoints[:, 4:5], rpoints[:, 5:6], rpoints[:, 6:7]
    axis.scatter(ix, iy, iz, c=c, s=s)
    
def plot_mpoints(axis, mpoints, c=None, s=None):
    ix, iy, iz = mpoints[:, 2:3], mpoints[:, 3:4], mpoints[:, 4:5]
    axis.scatter(ix, iy, iz, c=c, s=s)

def create_point_generator(vec_par, vec_offset):
    def generator(n, t):
        return vec_offset * n + vec_par * t
    return generator

def round2pow(n):
    return int(n**0.5)

def create_plain_manifold_generator(generate_point=None):
    scaling = 1
    def generator(n, t, atol, points_count=40):
        size = round2pow(points_count)
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

### Generator for first plain or "A" manifold
a_norm = np.array([1, 1, 1]) # Norm vector of the plane
a_par = np.array([1, -0.5, -0.5]) # Some orthogonal to norm vector
a_offset = np.array([0, 1, -1]) # One another orthogonal to previous other vectors
a_manifold_generator = create_plain_manifold_generator(create_point_generator(a_par, a_offset))

### Generator for second plain or "B" manifold
b_norm = np.array([1, 0, 1]) # Norm vector of the plane
b_par = np.array([1, 0, -1]) # Some orthogonal to norm vector
b_offset = np.array([0, 1, 0]) # One another orthogonal to previous other vectors
b_manifold_generator = create_plain_manifold_generator(create_point_generator(b_par, b_offset))

# Create visible planes
ax, ay, az = create_plain(a_norm)
bx, by, bz = create_plain(b_norm)

# Generate initial minifolds
a = a_manifold_generator(0, 0, 4)
b = b_manifold_generator(0, 0, 4)

# Find intersection
_, rpoints, _ = pyi.optimize3d(a, b, 0.1, a_manifold_generator, b_manifold_generator, max_iter=3)

# Plot results
fig = plt.figure(figsize=(10, 10))
axis = fig.add_subplot(111, projection='3d')
axis.plot_surface(ax, ay, az, alpha=0.5)
axis.plot_surface(bx, by, bz, alpha=0.5)
plot_rpoints(axis, rpoints, c='deeppink', s=20)
plt.show()


# plot_mpoints(axis, a, c='g', s=30)
# plot_mpoints(axis, b, c='b', s=30)


