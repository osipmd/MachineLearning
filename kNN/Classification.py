from kNN import *
from Statistics import *

class Classification:
    def __init__(self):
        self.units = []

    def add_unit(self, unit):
        self.units.append(unit)

    def calc_f_measure(self):
        f_measure = 0
        for unit in self.units:
            f_measure += unit.calculate_f_measure()
        return f_measure / len(self.units)

class ClassificationUnit:
    def __init__(self, train, test, classified):
        self.train = train
        self.test = test
        self.classified = classified

    def calculate_confusion_matrix(self):
        matrix = [[0, 0], [0, 0]]
        for i in range(len(self.classified)):
            fact = self.test[i]
            predicted = self.classified[i]

            # predicted for row in the matrix
            # and fact for column
            matrix[predicted.class_number][fact.class_number] += 1
        return matrix

    def calculate_f_measure(self):
        return Statistics.count_f_measure(self.calculate_confusion_matrix())
