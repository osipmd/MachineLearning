import os

from l2_utils import *
from analitic_regression import *
from gradient_descent import *
from l2_drawer import *

flats, max_area, max_rooms, max_price = normalize(read_data())


def analyse_coeffs(flats, coeffs, k=None, eps=None):
    model = create_model(coeffs)
    y = list(map(lambda flat: flat.price, flats))
    predicted_y = list(map(lambda flat: model(flat), flats))
    error = rms_error(y, predicted_y)
    with open('out/error.txt', 'a') as f:
        f.write("eps : " + str(eps) + "\n")
        f.write("k : " + str(k) + "\n")
        f.write("weights : \n")
        f.write("\tarea : " + str(coeffs[0]) + "\n")
        f.write("\trooms : " + str(coeffs[1]) + "\n")
        f.write("\tw_0 : " + str(coeffs[2]) + "\n")
        f.write("RMS : " + str(error) + "\n")
        f.write("sqrt(RMS) : " + str(math.sqrt(error)) + "\n")
        f.write("\n")


def count_analytic(flats):
    coeffs = AnalyticRegression.calc(flats)
    analyse_coeffs(flats, coeffs)


def count_gradient_descent(flats, eps=0.001, k=0.0000001):
    gd = GradientDescent()
    coeffs = gd.calc(flats, eps, k)
    analyse_coeffs(flats, coeffs, eps, k)
    # errors = step_error(gd.all_coeffs, flats)
    # Drawer().draw_step_error(errors[1:])


def count_gradient_descent_by_hand(flats):
    gd = GradientDescent()
    # to use had-written gradient calc function
    coeffs = gd.calc(flats, by_hand=True)
    analyse_coeffs(flats, coeffs)
    # errors = step_error(gd.all_coeffs, flats)
    # Drawer().draw_step_error(errors[1:])


if not os.path.exists("out"):
    os.makedirs("out")
with open('out/error.txt', 'w') as f:
    f.write("")

x, y = read_data_to_two_set()
x = (x - x.mean()) / x.std()

gd = GradientDescent()
coeffs = gd.calc_normalize(x, y)
print(coeffs)
print(math.sqrt(get_error(x, y, coeffs)))

