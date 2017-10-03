import numpy as np
from matplotlib import pyplot
from matplotlib.colors import ListedColormap
from kNN_count import *
from Point import *



class Drawer:
    @staticmethod
    def draw_data(data, result=False, number=0):
        classColormap = ListedColormap(['#00FF00', '#FF0000'])
        pyplot.scatter([data[i].x for i in range(len(data))],
                       [data[i].y for i in range(len(data))],
                       c=[data[i].class_number for i in range(len(data))],
                       cmap=classColormap)
        figure = pyplot.gcf()
        if result == True:
            title = 'Result data #' + str(number)
            figure.canvas.set_window_title(title)
        else:
            figure.canvas.set_window_title('Source data')
        pyplot.show()

    @staticmethod
    def draw_graphic(data):
        x = []
        for i in range(len(data)):
            x.append(i + 2)
        pyplot.bar(x, data)
        pyplot.show()

    @staticmethod
    def showDataOnMesh(source_data, result):
        def generateTestMesh(mesh_data):
            x_min = min([mesh_data[i].x for i in range(len(mesh_data))]) - 1.0
            x_max = max([mesh_data[i].x for i in range(len(mesh_data))]) + 1.0
            y_min = min([mesh_data[i].y for i in range(len(mesh_data))]) - 1.0
            y_max = max([mesh_data[i].y for i in range(len(mesh_data))]) + 1.0
            h = 0.05
            testX, testY = np.meshgrid(np.arange(x_min, x_max, h),
                                       np.arange(y_min, y_max, h))
            return [testX, testY]

        testMesh = generateTestMesh(result)
        test_data = []
        x = zip(testMesh[0].ravel(), testMesh[1].ravel())
        for i in x:
            test_data.append(Point(i[0], i[1], 0))
        testMeshLabels = kNN_count.classifyKNN(result, test_data)
        class_array = []
        for point in testMeshLabels:
            class_array.append(point.class_number)
        classColormap = ListedColormap(['#00FF00', '#FF0000'])
        testColormap = ListedColormap(['#AAFFAA', '#FFAAAA'])
        pyplot.pcolormesh(testMesh[0],
                          testMesh[1],
                          np.asarray(class_array).reshape(testMesh[0].shape),
                          cmap=testColormap)
        pyplot.scatter([source_data[i].x for i in range(len(source_data))],
                       [source_data[i].y for i in range(len(source_data))],
                       c=[source_data[i].class_number for i in range(len(source_data))],
                       cmap=classColormap)
        pyplot.show()
