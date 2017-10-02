from Point import *
from Statistics import *
from Drawer import *
import numpy as np


class k_Nearest_Neighbor:
    def __init__(self, number_of_classes, number_cross_validation, number_k_Neighbor, dist, kernel, data):
        self.number_of_classes = number_of_classes
        self.number_cross_validation = number_cross_validation
        self.number_k_Neighbor = number_k_Neighbor
        self.data = data
        self.kernel = kernel
        self.dist = dist

    def calculate_classification(self):
        for i in range(self.number_cross_validation):
            train_data, test_data = self.split_data_set(i)
            result = self.kNN_classifier(train_data, test_data)
            data = []
            for point in train_data:
                data.append(point)
            for point in result:
                data.append(point)
            Drawer.draw_data(data, True, i)
            print("F_measure: ", Statistics.count_f_measure(result, test_data))

    def kNN_classifier(self, train_data, test_data):
        test_result = []
        train_data_size = len(train_data)
        for test_point in test_data:
            distances = [self.dist(test_point, train_data[i]) for i in range(train_data_size)]
            neighbours = [[distances[i], train_data[i].class_number] for i in range(train_data_size)]
            sorted_neighbours = sorted(neighbours)
            similarity = [0] * self.number_of_classes
            for neighbour in sorted_neighbours[0:self.number_k_Neighbor]:
                class_of_neighbour = neighbour[1] 
                similarity[class_of_neighbour] += self.kernel(neighbour[0] / sorted_neighbours[self.number_k_Neighbor + 1][0])
            similarity_number_of_class = zip(similarity, range(self.number_of_classes))
            test_result_point = Point(test_point.x, test_point.y, sorted(similarity_number_of_class, reverse=True)[0][1])
            test_result.append(test_result_point)
        return test_result

    # split data using parameters
    # note: now 3 last elements always in train_data
    def split_data_set(self, shift):
        data_size = len(self.data)
        split_size = int(data_size / self.number_cross_validation)
        start = shift * split_size
        end = start + split_size
        train_data = self.data[0:start] + self.data[end:len(self.data)]
        test_data = self.data[start:end]
        return train_data, test_data
