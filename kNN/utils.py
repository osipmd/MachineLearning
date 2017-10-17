from kNN import *

def print_classification(classification):
    for unit in classification.units:
        print(unit)
        for point in unit.test:
            print(point, end=" , ")
        print()
        for point in unit.classified:
            print(point, end=" , ")
        print()
        print(unit.calculate_confusion_matrix())
        matrix = unit.calculate_confusion_matrix()
        print(Statistics.count_recall(matrix))
        print(Statistics.count_precision(matrix))
        print(Statistics.count_f_measure(matrix))
    print('f_measure ', classification.calc_f_measure())

# read data from file
def read_data_from_file(filename):
    data = []
    with open(filename) as input_file:
        for line in input_file:
            point = Point2D.from_str(line)
            data.append(point)
    return data

