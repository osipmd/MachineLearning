from flat import Flat
from l2_utils import create_model

w_0 = 89597.9095428

# area weight
w_1 = 139.21067402

# rooms weight
w_2 = -8738.01911233

coeffs = [w_1, w_2, w_0]
model = create_model(coeffs)

while True:
    area = int(input("area : "))
    rooms = int(input("rooms : "))
    flat = Flat(area, rooms)
    print("predicted price : ", int(model(flat) * 100) / 100)
    print()
