import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        pass
    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)