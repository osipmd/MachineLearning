from Point import Point


class k_Nearest_Neighbor:
    dataset_file = 'chips.txt'
    number_of_classes = 2
    number_cross_validation = 5
    number_k_Neighbor = 10

    # read data from file
    def read_data_from_file(dataset_file=dataset_file):
        data = []
        with open(dataset_file) as input_file:
            for line in input_file:
                x = line.split(",")[0]
                y = line.split(",")[1]
                class_number = line.split(",")[2]
                point = Point(float(x), float(y), int(class_number))
                data.append(point)
        return data

    def kNN_classifier(train_data, test_data, k, number_of_classes=number_of_classes):
        test_result = []
        for test_point in test_data:
            test_dist = [[get_Minkowski_distance(test_point, train_data[i]), train_data[i].class_number] for i in
                         range(len(train_data))]
            stat = [0 for i in range(number_of_classes)]
            sorted_dist = sorted(test_dist)
            for d in sorted_dist[0:k]:
                stat[d[1]] += get_Epanechnikov_kernel(d[0] / sorted_dist[k + 1][0])
            test_result.append(sorted(zip(stat, range(number_of_classes)), reverse=True)[0][1])
        return test_result

    # split data using parameters
    # note: now 3 last elements always in train_data
    @staticmethod
    def split_data_set(data, start, size):
        train_data = []
        test_data = []
        for i in range(len(data)):
            if start <= i < start + size:
                test_data.append(data[i])
            else:
                train_data.append(data[i])
        return train_data, test_data

    def calculate_classification(self):
        data = self.read_data_from_file()
        data_size = len(data)
        split_size = int(data_size / self.number_cross_validation)
        for i in range(self.number_cross_validation):
            train_data, test_data = k_Nearest_Neighbor.split_data_set(data, (i * split_size), split_size)
            result = self.kNN_classifier(train_data, test_data, self.number_k_Neighbor)
            print("F_measure: ", k_Nearest_Neighbor.count_f_measure(result, test_data))

    @staticmethod
    def count_tp(result, test_data):
        count = 0
        for i in range(len(result)):
            if result[i] == 1 and test_data[i].class_number == 1:
                count += 1
        return count

    @staticmethod
    def count_fn(result, test_data):
        count = 0
        for i in range(len(result)):
            if result[i] == 0 and test_data[i].class_number == 1:
                count += 1
        return count

    @staticmethod
    def count_fp(result, test_data):
        count = 0
        for i in range(len(result)):
            if result[i] == 1 and test_data[i].class_number == 0:
                count += 1
        return count

    @staticmethod
    def count_tn(result, test_data):
        count = 0
        for i in range(len(result)):
            if result[i] == 0 and test_data[i].class_number == 0:
                count += 1
        return count

    @staticmethod
    def count_recall(result, test_data):
        tp = k_Nearest_Neighbor.count_tp(result, test_data)
        fn = k_Nearest_Neighbor.count_fn(result, test_data)
        if tp == 0:
            return 0
        return tp / (tp + fn)

    @staticmethod
    def count_precision(result, test_data):
        tp = k_Nearest_Neighbor.count_tp(result, test_data)
        fp = k_Nearest_Neighbor.count_fp(result, test_data)
        if tp == 0:
            return 0
        return tp / (tp + fp)

    @staticmethod
    def count_f_measure(result, test_data, beta=1):
        precision = k_Nearest_Neighbor.count_precision(result, test_data)
        recall = k_Nearest_Neighbor.count_recall(result, test_data)
        if precision == 0 or recall == 0:
            return 0
        else:
            return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)


# We can use different distance. Default - Euclidean distance
def get_Minkowski_distance(a, b, p=2):
    dist = (abs((b.x - a.x) ** p) + abs((b.y - a.y) ** p)) ** (1 / p)
    return dist


# https://en.wikipedia.org/wiki/Kernel_(statistics)
def get_Epanechnikov_kernel(value):
    return 0.75 * (1 - value * value)


kNN = k_Nearest_Neighbor
kNN.calculate_classification(kNN)
