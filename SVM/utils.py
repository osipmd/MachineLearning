import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler


def read_data_from_file(file_name):
    data = pd.read_csv(file_name, header=None, names=['X', 'Y', 'Class'])
    data = data.sample(frac=1, random_state=0)
    X = data[['X', 'Y']].values
    Y = data['Class'].values
    data['R'] = (data['X'] ** 2 + data['Y'] ** 2) ** 0.5
    data['Φ'] = pd.np.arctan2(data['X'], data['Y'])
    X = data[['R', 'Φ']].values
    # X = data[['X', 'Y']].values
    y = data['Class'].values
    #class_colormap = ListedColormap(['#00FF00', '#FF0000'])
    #plt.scatter(x = data['X'], y = data['Y'],c = data['Class'], cmap=class_colormap)
    #plt.show()
    return X, Y


X, Y = read_data_from_file('chips.txt')
