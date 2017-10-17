from kNN import *
from Classification import *
from Point3D import *


class K_Nearest_Neighbor_3D:
    def __init__(self, number_of_classes, number_cross_validation, number_k_Neighbor, dist, kernel, data):
        self.number_of_classes = number_of_classes
        self.number_cross_validation = number_cross_validation
        self.k_neighbours = number_k_Neighbor
        self.data = data
        self.kernel = kernel
        self.dist = dist

    def calculate_classification(self):
        results = Classification()
        for i in range(self.number_cross_validation):
            train_data, test_data = self.split_data_set(i)
            classified_data = self.__classify(train_data, test_data)
            result = ClassificationUnit(train_data, test_data, classified_data)
            results.add_unit(result)
        return results

    # returns points from test_data,
    # which has class defined by kNN
    # accordingly to train_data
    def __classify(self, train_data, test_data):
        test_result = []
        train_data_size = len(train_data)
        for test_point in test_data:
            distances = [self.dist(test_point, train_data[i]) for i in range(train_data_size)]
            neighbours = [[distances[i], train_data[i].class_number] for i in range(train_data_size)]
            sorted_neighbours = sorted(neighbours)
            similarity = [0] * self.number_of_classes
            for neighbour in sorted_neighbours[0:self.k_neighbours]:
                class_of_neighbour = neighbour[1]
                similarity[class_of_neighbour] += self.kernel(
                    neighbour[0] / sorted_neighbours[self.k_neighbours + 1][0])

            similarity_and_number_of_class = zip(similarity, range(self.number_of_classes))
            the_most_similar_class = sorted(similarity_and_number_of_class, reverse=True)[0][1]

            test_result_point = Point3D(test_point.x, test_point.y, test_point.z, the_most_similar_class)
            test_result.append(test_result_point)
        return test_result

    # split data into two datasets: train and test
    # test_block_number is a number of block, which has to be test block
    def split_data_set(self, test_block_number):
        data_size = len(self.data)
        test_block_size = int(data_size / self.number_cross_validation)
        start = test_block_number * test_block_size
        end = start + test_block_size
        train_data = self.data[0:start] + self.data[end:data_size]
        test_data = self.data[start:end]
        return train_data, test_data
