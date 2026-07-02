from statistics import mean
from matplotlib import style
import numpy as np
import matplotlib.pyplot as plt

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6],dtype=np.float64)
ys = np.array([5,4,6,5,6,7],dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
    m = ( ((mean(xs)*mean(ys)) - mean(xs*ys)) /
          ((mean(xs)*mean(xs)) - mean(xs*xs)) )
    b = mean(ys) - m*mean(xs)
    return m, b
    # return m

m, b = best_fit_slope_and_intercept(xs,ys)
# b = best_fit_slope_and_intercept(xs,ys)[1]
print(m,b)
regression_line = [(m*x)+b for x in xs]

predict_x = 8
predict_y = (m*predict_x)+b


plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color='g')
plt.plot(xs, regression_line)
plt.show()

# plt.plot(xs, ys)
# plt.show()