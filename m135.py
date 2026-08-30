import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv('titanic.xls')
print(df.head())
