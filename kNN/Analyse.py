import random
from kNN import *
from Point2D import *
from Classification import *
from transformations import *
from Drawer import *
from distances import *
from kernels import *
from utils import *


number_of_classes = 2

data = read_data_from_file('chips.txt')

# data = data_to_polar(data)
# data = mult_data(data, 2)
random.shuffle(data)

best_number_k = 0
best_f_measure = 0

best_dist = distances.get('euclidian')
best_dist_name = ''

best_kernel = kernels.get('epanechnikov')
best_kernel_name = ''

best_cross_validation_number = 0

build_graphics = True
f_measure_of_results = []

for kernel_name, kernel in kernels.items():
    for dist_name, dist in distances.items():
        for cross_validation_number in range(5, 10):
            # file_name = "out_data/f-measure/k_results_" + str(cross_validation_number) + ".png"
            f_measure_of_results = []
            for number_k_Neighbor in range(5, 10):
                kNN = k_Nearest_Neighbor(number_of_classes, cross_validation_number, number_k_Neighbor, dist,
                                         kernel,
                                         data)
                classification = kNN.calculate_classification()
                cur_f_measure = classification.calc_f_measure()
                f_measure_of_results.append(cur_f_measure)

                if cur_f_measure > best_f_measure:
                    best_f_measure = cur_f_measure
                    best_number_k = number_k_Neighbor
                    best_dist = dist
                    best_dist_name = dist_name
                    best_kernel = kernel
                    best_kernel_name = kernel_name
                    best_cross_validation_number = cross_validation_number

            # if build_graphics:
            #     Drawer.draw_graphic(file_name, f_measure_of_results, 5)

kNN = k_Nearest_Neighbor(number_of_classes, best_cross_validation_number, best_number_k, best_dist,
                         best_kernel,
                         data)
classification = kNN.calculate_classification()

# if build_graphics:
#     Drawer.define_class_of_space(classification)

print('best k', best_number_k)
print('best dist ', best_dist_name)
print('best kernel ', best_kernel_name)
print('best cross validation number ', best_cross_validation_number)
print('f_measure ', classification.calc_f_measure())