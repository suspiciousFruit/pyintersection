import numpy as np
import orbipy as op
from tqdm import tqdm
from scipy.interpolate import splprep, splev
from scipy.optimize import brent
from scipy.spatial.distance import cdist


def create_manifold_generator(s0,
                              dstrb=3.355020244215249e-05,
                              grid=8):
    model = op.crtbp3_model()
    udir = op.unstable_direction_stm(op.crtbp3_model(stm=True))

    def generator(n0, n1, t0, t1, grid=(grid, grid)):
        """
        Generate manifold mesh
        """
        ns = np.linspace(n0, n1, grid[0])
        ts = np.linspace(t0, t1, grid[1])
        res = np.zeros((grid[1], grid[0], 5))
        for i, n in enumerate(tqdm(ns)):
            s = model.prop(s0, 0., n, ret_df=False)[-1, 1:]
            s[3:] += udir(n, s) * dstrb
            t0 = ts[0] - (t1 - t0) * 1e-5
            s1 = model.prop(s, 0., t0, ret_df=False)[-1, 1:]
            for j, t in enumerate(ts):
                s2 = model.prop(s1, t0, t, ret_df=False)[-1, 1:]
                res[j, i] = n, t, *s2[:3]
        return res  #.reshape(-1, 5)

    return generator


def create_yz_plane_generator(x):
    def yz_plane_generator(n0, n1, t0, t1, grid=(10, 10)):
        """
        YZ-plane mesh generator
        """
        py = np.linspace(n0, n1, grid[0])
        pz = np.linspace(t0, t1, grid[1])
        PY, PZ = np.meshgrid(py, pz)
        res = np.zeros((grid[1], grid[0], 5))
        res[..., 0] = PY
        res[..., 1] = PZ
        res[..., 2] = x
        res[..., 3] = res[..., 0]
        res[..., 4] = res[..., 1]
        return res

    return yz_plane_generator


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


def get_bbox(*arrs):
    arr = np.vstack(arrs)
    return arr.max(axis=0) - arr.min(axis=0)


def scale_interval(x0, x1, s):
    c = (x1 + x0) / 2
    d = (x1 - x0) / 2
    return c - d * s, c + d * s


def calc_nt_intervals(rpoints, m=0, s=1):
    mask = rpoints[:,1] == m
    intervals = []
    for l in np.unique(rpoints[mask, 0]):
        rpts = rpoints[mask][rpoints[mask, 0] == l]
        nmin, tmin = rpts[:, 2:4].min(axis=0)
        nmax, tmax = rpts[:, 2:4].max(axis=0)
        nmin, nmax = scale_interval(nmin, nmax, s)
        tmin, tmax = scale_interval(tmin, tmax, s)
        intervals.append([nmin, nmax, tmin, tmax])
    return np.array(intervals)


def generate_points(gen, intervals, grid=(8, 8)):
    pts = []
    tol = 0
    for intr in intervals:
        p = gen(*intr, grid)
        v = (p[1:, 1:, -3:] - p[:-1, :-1, -3:]).reshape(-1, 3)
        d = np.linalg.norm(v, axis=1)
        tol = max(tol, d.max())
        pts.append(gen(*intr, grid).reshape(-1, 5))
    return np.vstack(tuple(pts)), tol


def fit_spline(rpoints, m=0, k=3, s=0, n=1000):
    pts = rpoints[rpoints[:,1] == m]

    centers = []
    for cid in np.unique(pts[:, 0]):
        rpts = pts[pts[:, 0] == cid, 2:4]
        centers.append(rpts.mean(axis=0))
    centers = np.array(centers)
    D = cdist(centers, centers)  # distances between cluster centers
    start = np.unravel_index(np.argmax(D), D.shape)  # index of starting cluster center
    np.fill_diagonal(D, np.inf)
    start_node = start[1]
    path, dist = build_path(D.copy(), start_node)
    if dist.max() > 2 * dist.mean():
        start_node = path[np.argmax(dist)]
        path, dist = build_path(D.copy(), start_node)

    rough_tck, rough_u = splprep(centers[path].T, s=s, k=k)
    uu = np.linspace(rough_u.min(), rough_u.max(), n)
    return splev(uu, rough_tck)
