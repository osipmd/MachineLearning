import numpy as np
import matplotlib.pyplot as plt

class Drawer:
    def draw_step_error(self, errors):
        t = np.arange(0., len(errors))
        plt.plot(t, errors, 'r')
        plt.show()