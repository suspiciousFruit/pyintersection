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


# Кубы сожержат все пересечения



def cube_has_point(cube, point):
    pass

def cube_has_points(cube, points, manifold):
    pass



def test__1(cubes, points):
    for cube in cubes:
        a = cube_has_points(cube, points, 0)
        b = cube_has_points(cube, points, 1)
        if not a or not b:
            return False
    return True

