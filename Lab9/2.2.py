# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#Evaluating KNeighborsClassifier from k=1 to 20
#and record the classification accuracy
k_range = range(1,21)
KNeighbors_accuracy = []
for k in k_range:
    KNeighbors_model = KNeighborsClassifier(n_neighbors=k)
    KNeighbors_model.fit(x_train,y_train)
    KNeighbors_predictions=KNeighbors_model.predict(x_test)
    KNeighbors_accuracy.append(accuracy_score(y_test,KNeighbors_predictions))
#plot the relationship between K and the classification accuracy
plt.plot(k_range,KNeighbors_accuracy)
plt.xlabel('Value of K for KNN')
plt.ylabel('Classification Accuracy')
plt.show()