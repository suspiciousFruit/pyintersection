# 1. Каждый куб содержит хотя бы по одной точке из каждого многообразия (Причем cid куба и cid точек совпадают)
# 2. Каждая точка попала хотя бы в один из кубов (Причем cid куба и cid точки совпадают)
# 3. Каждая точка попала лишь в один куб (Причем cid куба и cid точки совпадают)
# 
# 4. Каждая точка из ответа содержится в одном из многообразий (номер многообразия совпадает с номером точки)
# 
#
#
#
#
import numpy as np

def rpoints_equal(p1, p2):
    return np.all(p1 == p2)

def points_equal_rm(rpoint, mpoint):
    return np.all(rpoint[2:] == mpoint)

class Tests:
    def __init__(self, dims=None):
        def chp(cube, point):
            c = cube[1:]
            p = point[4:]
            if cube[0] != point[0]:
                return False
            for i in range(dims):
                if c[2*i] > p[i] or p[i] > c[2*i+1]:
                    return False
            return True
        
        self.cube_has_point = chp
        self.dims = dims
    
    def cube_has_both_points(self, cube, points):
        m0, m1 = False, False
        for point in points:
            if self.cube_has_point(cube, point):
                if not m0 and point[1] == 0:
                    m0 = True
                if not m1 and point[1] == 1:
                    m1 = True
        return m0 and m1
    
    # Каждый куб содержит хотя бы по одной точке из каждого многообразия (1)
    def test__1(self, rcubes, rpoints, manifolds):
        for rcube in rcubes:
            if not self.cube_has_both_points(rcube, rpoints):
                return False
        return True
    
    # Каждый куб не содержит точек, которые ему подходят по coords, но cid неверный (2)
    def test__2(self, rcubes, rpoints, manifolds):
        for rcube in rcubes:
            for rpoint in rpoints:
                if self.cube_has_point(rcube, rpoint) and rcube[0] != rpoint[0]:
                    return False
        return True
    
    # Каждый куб не содержит точек, которые имеют его cid, но coords неверные (3)
    def test__3(self, rcubes, rpoints, manifolds):
        for rcube in rcubes:
            for rpoint in rpoints:
                if rcube[0] == rpoint[0] and not self.cube_has_point(rcube, rpoint):
                    return False
        return True
    
    def point_is_in_only_one_cube(self, rcubes, rpoint):
        num = 0
        for rcube in rcubes:
            if self.cube_has_point(rcube, rpoint):
                num += 1
        return num == 1

    # Каждая точка попала хотя бы в один из кубов (Причем cid куба и cid точки совпадают)
    # Каждая точка попала лишь в один куб (Причем cid куба и cid точки совпадают) (4)
    def test__4(self, rcubes, rpoints, manifolds):
        for rpoint in rpoints:
            if not self.point_is_in_only_one_cube(rcubes, rpoint):
                return False
        return True
    
    # Кубы не пересекаются (5)
    def test__5(self, rcubes, rpoints, manifolds):
        return True
    
    # Стороны всех кубов корректны (6)
    def test__6(self, rcubes, rpoints, manifolds):
        for rcube in rcubes:
            c = rcube[1:]
            for i in range(self.dims):
                if c[2*i] > c[2*i+1]:
                    return False
            return True
    
    # В ответе точки только двух многообразий (7)
    def test__7(self, rcubes, rpoints, manifolds):
        s = set()
        for p in rpoints:
            s.add(p[1])
        return len(s) == 2
    
    def point_in_true_manifold(self, rpoint, mans):
        if rpoint[1] == 0:
            man = mans[0]
        elif rpoint[1] == 1:
            man = mans[1]
        else:
            return False
        for mpoint in man:
            if points_equal_rm(rpoint, mpoint):
                return True
        return False

    # Все точки из ответа находятся в своем начальном многообразии (8)
    def test__8(self, rcubes, rpoints, manifolds):
        for rpoint in rpoints:
            if not self.point_in_true_manifold(rpoint, manifolds):
                return False
        return True
    
    # В начальных данных нет точек, которые подходят кубам, но при этом отсутсвуют в ответе (9)
    def point_in_manifold(self, rpoint, man):
        for mpoint in man:
            if points_equal_rm(rpoint, mpoint):
                return True
        return False

    def mpoint_in_rpoints(self, mpoint, rpoints):
        for rpoint in rpoints:
            if points_equal_rm(rpoint, mpoint):
                return True
        return False

    def cube_fit_mpoint(self, cube, mpoint):
        c = cube[1:]
        for i in range(self.dims):
            if c[2*i] > mpoint[i] or mpoint[i] > c[2*i+1]:
                return False
        return True

    def test__9(self, rcubes, rpoints, mans):
        for man in mans:
            for mpoint in man:
                for cube in rcubes:
                    if self.cube_fit_mpoint(cube, mpoint) and not self.mpoint_in_rpoints(mpoint, rpoints):
                        return False
        return True

    def run(self, rcubes, rpoints, manifolds):
        number = 1
        while hasattr(self, f'test__{number}'):
            test = getattr(self, f'test__{number}')
            print(f'test {number}', test(rcubes, rpoints, manifolds))
            number += 1
