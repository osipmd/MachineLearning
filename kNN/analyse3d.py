import random
from knn_3d import *
from Point3D import *
from Classification import *
from transformations import *
from Drawer import *
from distances import *
from kernels import *
from utils import *

number_of_classes = 2

data = read_data_from_file('chips.txt')

random.shuffle(data)

arr = []

for i in range(1, 5):
    arr.append(i / 10)

print(arr)

# best_a = 0.5
# best_b = 0.5

for a in arr:
    # print(a)
    for b in arr:
        # print(b)
        data = data_to_elliptic(data, a, b)

        best_number_k = 0
        best_f_measure = 0

        best_dist = distances_3d.get('euclidian')
        best_dist_name = ''

        best_kernel = kernels.get('epanechnikov')
        best_kernel_name = ''

        best_cross_validation_number = 0

        f_measure_of_results = []



        # data = data_to_hyperbolic(data, a, b)
        for kernel_name, kernel in kernels.items():
            for dist_name, dist in distances_3d.items():
                for cross_validation_number in range(5, 20):
                    f_measure_of_results = []
                    for number_k_Neighbor in range(5, 20):
                        kNN = K_Nearest_Neighbor_3D(number_of_classes, cross_validation_number, number_k_Neighbor, dist,
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
                            best_a = a
                            best_b = b
        kNN = K_Nearest_Neighbor_3D(number_of_classes, best_cross_validation_number, best_number_k, best_dist,
                                    best_kernel,
                                    data)
        classification = kNN.calculate_classification()

        print('best k', best_number_k)
        print('best dist ', best_dist_name)
        print('best kernel ', best_kernel_name)
        print('best cross validation number ', best_cross_validation_number)
        print('f_measure ', classification.calc_f_measure())
        print('best_a ', best_a)
        print('best_b ', best_b)
        print()
        print()
        print('--------------------------')
        print()
        print()

