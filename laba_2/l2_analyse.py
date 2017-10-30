import math

from l2_utils import *
from analitic_regression import *
from gradient_descent import *

flats = read_data()

# coeffs = AnalyticRegression.calc(flats)
coeffs = GradientDescent.calc(flats)

# to use had-written gradient calc function
# coeffs = GradientDescent.calc(flats, True)

print(coeffs)

model = create_model(coeffs)

y = list(map(lambda flat: flat.price, flats))
predicted_y = list(map(lambda flat: model(flat), flats))
#
error = rms_error(y, predicted_y)
print("RMS : ", error)
print("sqrt(RMS) : ", math.sqrt(error))

# while True:
#     area = int(input("area : "))
#     rooms = int(input("rooms : "))
#     flat = Flat(area, rooms)
#     print("predicted price : ", model(flat))
