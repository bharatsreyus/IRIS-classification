import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

#loading the data
iris= load_iris()

features = iris.data.T

sepal_length = features[0]
sepal_width = features[1]
petal_length = features[2]
petal_width = features[3]

sepal_length_name = iris.feature_names[0]
sepal_width_name = iris.feature_names[1]
petal_length_name = iris.feature_names[2]
petal_width_name = iris.feature_names[3]

#visualising the data
plt.scatter(sepal_length,sepal_width, c = iris.target )
plt.xlabel(sepal_length_name)
plt.ylabel(sepal_width_name)
plt.show()

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
#ML algorithm
knn = KNeighborsClassifier(n_neighbors = 1)
#fitting the training data
knn.fit(X_train, y_train)

#print score
score = knn.score(X_test,y_test)
print("Score of the model: ", score)


#input in the form of np array
print("Enter the values: ", end=" ")
l=list(map(float,input().split()))
X_new = np.array([l])

#prediction
prediction = knn.predict(X_new)

#output
if prediction[0]==0:
    print("SETOSA")
elif prediction[0]==1:
    print("VERSICOLOR")
else :
    print("VIRGINICA")
