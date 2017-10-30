class Flat:
    def __init__(self, area, rooms, price=0):
        self.area = area
        self.rooms = rooms
        self.price = price


    def get_attrs_list(self):
        w_0 = 1
        return [self.area, self.rooms, w_0]