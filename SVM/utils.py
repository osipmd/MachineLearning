import pandas as pd

def read_data_from_file(file_name):
    data = pd.read_csv(file_name, header=None, names=['X', 'Y', 'Class'])
    data = data.sample(frac=1, random_state=0)
    X = data[['X', 'Y']].values
    Y = data['Class'].values
    #class_colormap = ListedColormap(['#00FF00', '#FF0000'])
    #plt.scatter(x = data['X'], y = data['Y'],c = data['Class'], cmap=class_colormap)
    #plt.show()
    return X, Y

