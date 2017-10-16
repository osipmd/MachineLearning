from Point import *

class kNN_for_background:
    @staticmethod
    def classifyKNN(train_data, test_data):
        epanechnikov_kernel = lambda val: 0.75 * (1 - val * val)
        minkowski_distance = lambda a, b, p: (abs((b.x - a.x) ** p) + abs((b.y - a.y) ** p)) ** (1 / p)
        test_result = []
        train_data_size = len(train_data)
        for test_point in test_data:
            distances = [minkowski_distance(test_point, train_data[i], 2) for i in range(train_data_size)]
            neighbours = [[distances[i], train_data[i].class_number] for i in range(train_data_size)]
            sorted_neighbours = sorted(neighbours)
            similarity = [0] * 2
            for neighbour in sorted_neighbours[0:10]:
                class_of_neighbour = neighbour[1]
                similarity[class_of_neighbour] += epanechnikov_kernel(
                    neighbour[0] / sorted_neighbours[10 + 1][0])
            similarity_number_of_class = zip(similarity, range(2))
            test_result_point = Point(test_point.x, test_point.y,
                                      sorted(similarity_number_of_class, reverse=True)[0][1])
            test_result.append(test_result_point)
        return test_result
