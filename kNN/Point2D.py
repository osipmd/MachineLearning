from kNN import *

class Point2D:
    def __init__(self, x, y, class_number):
        self.x = x
        self.y = y
        self.class_number = class_number

    @staticmethod
    def from_str(str):
        x, y, class_number = str.split(",")
        return Point2D(float(x), float(y), int(class_number))

    def __str__(self):
        return '{0} {1} {2}'.format(self.x, self.y, self.class_number)
