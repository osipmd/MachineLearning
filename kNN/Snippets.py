from KDTree import *
from PointKDTree import *
from Point import *

# points = [[1, 2], [2, 20], [5, 4], [3, 8], [2, 23], [5, 17], [32, 5], [32, 65]]
# tree = KDTree(points, 2)
# tree.print()
# print(tree.search([10, 22]))


points = [Point(1, 2, 1), Point(2, 20, 1), Point(5, 4, 1), Point(3, 8, 0), Point(2, 23, 1), Point(5, 17, 0),
          Point(32, 5, 1),
          Point(32, 65, 1)]
tree = PointKDTree(points, 2)
tree.print()
print(tree.search(Point(-1, -1, 1)))


# points = [Point(1.2, 2.3, 0)]
#
# point = Point(1.2, 2.3, 0)
