from l2_utils import *
from analitic_regression import *

points = read_data()

coeffs = AnalyticRegression.calc(points)

model = create_model(coeffs)

y = list(map(lambda point: point.price, points))
predicted_y = list(map(lambda point: model(point), points))

error = rms_error(y, predicted_y)

print(error)

