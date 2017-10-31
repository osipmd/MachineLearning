from l2_utils import *
from analitic_regression import *
from gradient_descent import *
from l2_drawer import *
flats = read_data()


# coeffs = AnalyticRegression.calc(flats)



gd = GradientDescent()
# coeffs = gd.calc(flats)
# to use had-written gradient calc function
coeffs = gd.calc(flats, by_hand=True)

print(coeffs)

model = create_model(coeffs)

y = list(map(lambda flat: flat.price, flats))
predicted_y = list(map(lambda flat: model(flat), flats))

error = rms_error(y, predicted_y)
print("RMS : ", error)
print("sqrt(RMS) : ", math.sqrt(error))

errors = step_error(gd.all_coeffs, flats)

Drawer().draw_step_error(errors)

# print(list(map(lambda error: math.sqrt(error), step_error(gd.all_coeffs, flats))))

# while True:
#     area = int(input("area : "))
#     rooms = int(input("rooms : "))
#     flat = Flat(area, rooms)
#     print("predicted price : ", model(flat))
