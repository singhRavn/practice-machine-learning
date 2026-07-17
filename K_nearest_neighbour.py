import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, linear_model, svm, neighbors
accuracies = []
for i in range(25):
    df = pd.read_csv('wdbc.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], axis=1, inplace=True)

    x = np.array(df.drop(['diagnosis'], axis=1))
    y= np.array(df['diagnosis'])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)

    accuracy = clf.score(x_test, y_test)
    # print(accuracy)

    # example_measures = np.array([[4,2,1,1,1,2,3,2,1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], [4,2,1,1,1,2,3,2,1,1,2,3,4,5,6,7,8,9,10,11,12,67,14,15,16,17,18,19,20,21]])
    # example_measures = example_measures.reshape(len(example_measures), -1)


    # prediction = clf.predict(example_measures)
    # print(prediction)
    accuracies.append(accuracy)
print(sum(accuracies)/len(accuracies))