import matplotlib.pyplot as plt


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