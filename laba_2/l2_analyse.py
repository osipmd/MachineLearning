from l2_utils import *
from analitic_regression import *

flats = read_data()

coeffs = AnalyticRegression.calc(flats)

model = create_model(coeffs)

y = list(map(lambda flat: flat.price, flats))
predicted_y = list(map(lambda flat: model(flat), flats))

error = rms_error(y, predicted_y)
print("RMS : ", error)

while True:
    area = int(input("area : "))
    rooms = int(input("rooms : "))
    flat = Flat(area, rooms)
    print("predicted price : ", model(flat))
