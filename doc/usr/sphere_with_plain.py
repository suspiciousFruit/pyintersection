import numpy as np
import pyintersection as pyi
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline



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



def create_sphere_mesh(r):
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
    return x, y, z



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

def create_point_generator(vec_par, vec_offset, D=np.array([0.0, 0.0, 0.0])):
    def generator(n, t):
        return vec_offset * n + vec_par * t + D
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
plain_manifold_generator = create_manifold_generator(plain_point_generator, points_count=80) # Manifold generator for plain
plain_mesh = create_plain_mesh(plain_norm) # Plain mesh

# Manifold generator for sphere
sphere_manifold_generator = create_manifold_generator(sphere_point_generator, points_count=80)
sphere_mesh = create_sphere_mesh(5.0)



# Create initial manifolds
plain = plain_manifold_generator(0, 0, 10)
sphere = sphere_manifold_generator(0, 0, 7)

# Find optimize intersection with abs tolerance 0.1
rcubes, rpoints, _ = pyi.optimize3d(plain, sphere, 0.1,
    plain_manifold_generator, sphere_manifold_generator, max_iter=10)



from sklearn.cluster import k_means
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev
from scipy.optimize import brent



def build_path(D, start):
    """
    Build path trough all nodes using minimum distance stepping
    :param D: distance matrix
    :param start: index of starting node
    :return: path
    """
    visited = [start]
    dist = [0]
    n = D.shape[0]
    while len(visited) < n:
        current = visited[-1]
        next = np.argmin(D[current])
        while next in visited:
            D[current, next] = np.inf
            D[next, current] = np.inf
            next = np.argmin(D[current])
        dist.append(D[current, next])
        D[current, next] = np.inf
        D[next, current] = np.inf
        visited.append(next)
    return np.array(visited), np.array(dist)



points = np.array([*filter(lambda x: x[1] == 0, rpoints)])[:, 2:4]
nclusters = 10
nt_centers, labels, _ = k_means(points, nclusters)



D = cdist(nt_centers, nt_centers)  # distances between cluster centers
start = np.unravel_index(np.argmax(D), D.shape)  # index of starting cluster center
np.fill_diagonal(D, np.inf)



start_node = start[1] # starting node choice need to be done clever
path, dist = build_path(D.copy(), start_node)

if dist.max() > 2 * dist.mean():
    start_node = path[np.argmax(dist)]
    path, dist = build_path(D.copy(), start_node)



rough_tck, rough_u = splprep(nt_centers[path].T, s=0, k=3)
add_pts = splev(np.linspace(-0.1, 1.1, 200), rough_tck)



vals = np.empty(points.shape[0])
for i in range(points.shape[0]):
    p = points[i]
    cidx = np.where(path == labels[i])[0][0]
    v0 = rough_u[cidx]
    v = brent(lambda l: np.sum((splev(l, rough_tck) - p) ** 2),
              brack=(v0 - 0.05, v0 + 0.05),
              tol=1e-5)
    vals[i] = v

idx = np.argsort(vals)



u = vals[idx]
interp_tck, _ = splprep(points[idx].T, u=u, k=3, s=0)
smooth_tck, _ = splprep(points[idx].T, u=u, k=3, s=0.3)
uu = np.linspace(u.min(), u.max(), 1000)
smooth_pts = splev(uu, smooth_tck)
interp_pts = splev(uu, interp_tck)



# Build points
points = np.array([plain_point_generator(smooth_pts[0][i], smooth_pts[1][i]) for i in range(len(smooth_pts[0]))])

# Plot results
fig = plt.figure(figsize=(12, 12))
axis = fig.add_subplot(111, projection='3d')
plot_surface(axis, plain_mesh, alpha=0.5)
plot_surface(axis, sphere_mesh, alpha=0.5)
plot_rawpoints(axis, points, c='g', s=4)
plt.show()
