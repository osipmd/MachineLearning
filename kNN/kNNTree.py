from Point import *
from PointKDTree import *


class kNNTree:
    def __init__(self, number_of_classes, number_cross_validation, number_k_Neighbor, dist, kernel, data):
        self.number_of_classes = number_of_classes
        self.number_cross_validation = number_cross_validation
        self.number_k_Neighbor = number_k_Neighbor
        self.data = data
        self.kernel = kernel
        self.dist = dist

    def calculate_classification(self):
        results = []
        for i in range(self.number_cross_validation):
            train_data, test_data = self.split_data_set(i)
            result = self.kNN_classifier(train_data, test_data)
            results.append([train_data, result, test_data])
        return results

    def kNN_classifier(self, train_data, test_data):
        test_result = []
        tree = PointKDTree(train_data, 10)
        for test_point in test_data:
            neighbours = tree.search(test_point)
            similarity = [0] * self.number_of_classes
            for neighbour in neighbours[0:self.number_k_Neighbor]:
                class_of_neighbour = neighbour.class_number
                similarity[class_of_neighbour] += self.kernel(
                    self.dist(test_point, neighbour) / [self.number_k_Neighbor + 1][0])
            similarity_number_of_class = zip(similarity, range(self.number_of_classes))
            test_result_point = Point(test_point.x, test_point.y,
                                      sorted(similarity_number_of_class, reverse=True)[0][1])
            test_result.append(test_result_point)
        return test_result

    def kNN_classifier(self, train_data, test_data):
        test_result = []
        tree = PointKDTree(train_data, 5)
        for test_point in test_data:
            point_neighbours = tree.search(test_point)
            neighbours_size = len(point_neighbours)
            distances = []
            for train_point in point_neighbours:
                distances.append(self.dist(test_point, train_point))
            neighbours = [[distances[i], point_neighbours[i].class_number] for i in range(neighbours_size)]

            sorted_neighbours = sorted(neighbours)

            similarity = [0] * self.number_of_classes
            for neighbour in sorted_neighbours[0:self.number_k_Neighbor]:
                class_of_neighbour = neighbour[1]
                similarity[class_of_neighbour] += self.kernel(
                    neighbour[0] / sorted_neighbours[self.number_k_Neighbor + 1][0])
            similarity_number_of_class = zip(similarity, range(self.number_of_classes))
            test_result_point = Point(test_point.x, test_point.y,
                                      sorted(similarity_number_of_class, reverse=True)[0][1])
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
