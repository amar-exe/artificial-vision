# import the necessary packages
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#Evaluating SVMClassifier for c=[1,5,10,100] and record classification accuracy
c_range = [1,5,10,100]
SVM_accuracy = []
for c in c_range:
    SVM_model = SVC(C=c,kernel='linear')
    SVM_model.fit(x_train,y_train)
    SVM_predictions=SVM_model.predict(x_test)
    SVM_accuracy.append(accuracy_score(y_test,SVM_predictions))
#plot the relationship between C and the classification accuracy
plt.scatter(c_range,SVM_accuracy)
plt.xlabel('Value of C for SVM')
plt.ylabel('Classification Accuracy')
plt.show()