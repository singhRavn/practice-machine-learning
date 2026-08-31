import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv('titanic.xls')
# df.drop(['ticket', 'fare', 'cabin', 'embarked', 'home.dest'], 1, inplace=True)
df.drop (['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(value=0, inplace=True)

print(df.head())