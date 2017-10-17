import numpy as np
from matplotlib import pyplot
from matplotlib.colors import ListedColormap
from kNN_for_background import *
from Point2D import *
from Classification import *


class Drawer:
    @staticmethod
    def draw_data(data, result=False, number=0):
        class_colormap = ListedColormap(['#00FF00', '#FF0000'])
        pyplot.scatter([data[i].x for i in range(len(data))],
                       [data[i].y for i in range(len(data))],
                       c=[data[i].class_number for i in range(len(data))],
                       cmap=class_colormap)
        figure = pyplot.gcf()
        if result == True:
            title = 'Result data #' + str(number)
            figure.canvas.set_window_title(title)
        else:
            figure.canvas.set_window_title('Source data')
        pyplot.show()

    @staticmethod
    def draw_graphic(file_name, data, start_k):
        x = []
        for i in range(len(data)):
            x.append(start_k + i)
        pyplot.bar(x, data, color='blue')
        pyplot.savefig(file_name)
        pyplot.close()

    @staticmethod
    def define_class_of_space(classification):
        #    def define_class_of_space(source_data, result):
        def generate_background_space(mesh_data):
            x_min = min([mesh_data[i].x for i in range(len(mesh_data))]) - 0.5
            x_max = max([mesh_data[i].x for i in range(len(mesh_data))]) + 0.5
            y_min = min([mesh_data[i].y for i in range(len(mesh_data))]) - 0.5
            y_max = max([mesh_data[i].y for i in range(len(mesh_data))]) + 0.5
            h = 0.05
            testX, testY = np.meshgrid(np.arange(x_min, x_max, h),
                                       np.arange(y_min, y_max, h))
            return [testX, testY]

        for j in range(len(classification.units)):
            unit = classification.units[j]
            source_data = unit.train + unit.test
            result = unit.train + unit.classified

            background_space = generate_background_space(result)
            background_data = []
            x = zip(background_space[0].ravel(), background_space[1].ravel())
            for i in x:
                background_data.append(Point2D(i[0], i[1], 0))
            result_labels = kNN_for_background.classifyKNN(result, background_data)
            class_array = []
            for point in result_labels:
                class_array.append(point.class_number)
            class_colormap = ListedColormap(['#00FF00', '#FF0000'])
            result_colormap = ListedColormap(['#AAFFAA', '#FFAAAA'])
            pyplot.pcolormesh(background_space[0],
                              background_space[1],
                              np.asarray(class_array).reshape(background_space[0].shape),
                              cmap=result_colormap)
            pyplot.scatter([source_data[i].x for i in range(len(source_data))],
                           [source_data[i].y for i in range(len(source_data))],
                           c=[source_data[i].class_number for i in range(len(source_data))],
                           cmap=class_colormap)
            file_name = "out_data/clustering-" + str(j + 1) + ".png"
            pyplot.savefig(file_name)
            pyplot.close()
