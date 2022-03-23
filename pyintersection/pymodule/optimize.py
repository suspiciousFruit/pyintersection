import numpy as np
import pyintersection as pyi



def get_manifold_number(rpoint):
    return rpoint[1]

def get_rpoint_params(rpoint):
    return rpoint[2], rpoint[3]

def generate_new_points(rpoints, atol=None, agen=None, bgen=None):
    ares, bres = [], []
    for rpoint in rpoints:
        n, t = get_rpoint_params(rpoint)
        manifold = get_manifold_number(rpoint)
        
        if manifold == 0:
            new_points = agen(n, t, atol)
            ares.append(new_points)
        elif manifold == 1:
            new_points = bgen(n, t, atol)
            bres.append(new_points)
        else:
            raise Exception("Wrong manifold number")
          
    counter = 0
    for nparr in ares:
        for unarr in nparr:
            counter += 1
    ares = np.reshape(np.array(ares), (counter, 5))
            
    counter = 0
    for nparr in bres:
        for unarr in nparr:
            counter += 1
    bres = np.reshape(np.array(bres), (counter, 5))
        
    return ares, bres

def optimize3d(a, b, target_atol=None, agen=None, bgen=None, max_iter=8):
    atol = max(pyi.get_boundary_cube3d(a, b))
    for i in range(max_iter):
        rcubes, rpoints, atols = pyi.intersect3d(a, b, atol=atol/2)
        #print(f'Iteration {i}: target_atols={atol/2} real_atols={max(atols)} rpoins={len(rpoints)} cubes={len(rcubes)}')
        if max(atols) <= target_atol:
            break
        atol = max(atols)
        a, b = generate_new_points(rpoints, atol=atol, agen=agen, bgen=bgen)
    return rcubes, rpoints, atols