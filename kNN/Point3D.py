from kNN import *


class Point3D:
    def __init__(self, x, y, z, class_number):
        self.x = x
        self.y = y
        self.z = z
        self.class_number = class_number

    def __str__(self):
        return '{0} {1} {2} {3}'.format(self.x, self.y, self.z, self.class_number)
