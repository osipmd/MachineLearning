class Point:
    def __init__(self, x, y, class_number):
        self.x = x
        self.y = y
        self.class_number = class_number

    @staticmethod
    def from_str(str):
        x, y, class_number = str.split(",")
        return Point(float(x), float(y), int(class_number))


