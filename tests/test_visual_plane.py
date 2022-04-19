import pyintersection as pyi
import numpy as np
import matplotlib.pyplot as plt

L = 100
W = 50
kx = 31
ky = 31

def make_xyz(n, D=0):
    n = n / np.linalg.norm(n)
    x, y = np.mgrid[-L:L:kx*1j, -W:W:ky*1j]
    A, B, C = n
    z = -(A*x+B*y+D)/C
    return (x, y, z)

def make_tnxyz_flatten(n, D=0):
    n = n / np.linalg.norm(n)
    x, y = np.mgrid[-L:L:kx*1j, -W:W:ky*1j]
    A, B, C = n
    z = -(A*x+B*y+D)/C
    size = z.shape[0] * z.shape[1]
    pack = zip(range(size), range(size), x.flatten(), y.flatten(), z.flatten())
    return np.array([np.array([t0, n0, x0, y0, z0]) for t0, n0, x0, y0, z0 in pack])

n1 = np.array([1, 1, 1])
n2 = np.array([1, 0, 1])
apoints = make_tnxyz_flatten(n1)
bpoints = make_tnxyz_flatten(n2)
ax, ay, az = make_xyz(n1)
bx, by, bz = make_xyz(n2)
cubes, points = pyi.intersect3d(apoints, bpoints, 0.5)
ix, iy, iz = points[:, 4:5], points[:, 5:6], points[:, 6:7]

fig = plt.figure(figsize=(10, 10))
axis = fig.add_subplot(111, projection='3d')
axis.plot_surface(ax, ay, az, alpha=0.5)
axis.plot_surface(bx, by, bz, alpha=0.5)
axis.scatter(ix, iy, iz)
plt.show()
