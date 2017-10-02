from Point import *
from Statistics import *


class k_Nearest_Neighbor:
    def __init__(self, number_of_classes, number_cross_validation, number_k_Neighbor, dist, kernel,
                 filename='chips.txt'):
        self.number_of_classes = number_of_classes
        self.number_cross_validation = number_cross_validation
        self.number_k_Neighbor = number_k_Neighbor
        self.data = self.read_data_from_file(filename)
        self.kernel = kernel
        self.dist = dist

    def calculate_classification(self):
        for i in range(self.number_cross_validation):
            train_data, test_data = self.split_data_set(i)
            result = self.kNN_classifier(train_data, test_data)
            print("F_measure: ", Statistics.count_f_measure(result, test_data))

    def kNN_classifier(self, train_data, test_data):
        test_result = []
        for test_point in test_data:
            test_dist = [[self.dist(test_point, train_data[i]), train_data[i].class_number] for i in
                         range(len(train_data))]
            stat = [0 for i in range(self.number_of_classes)]
            sorted_dist = sorted(test_dist)
            for d in sorted_dist[0:self.number_k_Neighbor]:
                stat[d[1]] += self.kernel(d[0] / sorted_dist[self.number_k_Neighbor + 1][0])
            test_result.append(sorted(zip(stat, range(self.number_of_classes)), reverse=True)[0][1])
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


    # read data from file
    @staticmethod
    def read_data_from_file(filename):
        data = []
        with open(filename) as input_file:
            for line in input_file:
                point = Point.from_str(line)
                data.append(point)
        return data
