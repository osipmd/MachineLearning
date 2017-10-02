from matplotlib import pyplot
from matplotlib.pyplot import *
from matplotlib.colors import ListedColormap


class Drawer:
    @staticmethod
    def draw_data(data, result=False, number=0):
        classColormap = ListedColormap(['#00FF00', '#FF0000'])
        pyplot.scatter([data[i].x for i in range(len(data))],
                       [data[i].y for i in range(len(data))],
                       c=[data[i].class_number for i in range(len(data))],
                       cmap=classColormap)
        figure = pyplot.gcf()
        if (result == True):
            title = 'Result data #' + str(number)
            figure.canvas.set_window_title(title)
        else:
            figure.canvas.set_window_title('Source data')
        pyplot.show()
