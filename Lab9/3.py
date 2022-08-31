# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import joblib # for old versions use: from sklearn.externals import joblib
# Load BCW dataset
BCW_dataset = load_breast_cancer()
x= BCW_dataset.data
y= BCW_dataset.target
numFolds=5
skf = StratifiedKFold(n_splits=numFolds,shuffle=True,random_state=0)
model_folds_accuracy = []
best_model_accuracy =0
best_model =[]
x_test_best_model=[]
y_test_best_model=[]
for train_fold_indexs, test_fold_indexs in skf.split(x, y):
    x_train_fold = x[train_fold_indexs];
    y_train_fold = y[train_fold_indexs];
    x_test_fold = x[test_fold_indexs];
    y_test_fold = y[test_fold_indexs];
# train the model using the k fold train data
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train_fold,y_train_fold)
# make predictions using the k fold test data
    predictions = model.predict(x_test_fold)
# compute accuracy for the testing set of the k fold
    fold_accuracy= accuracy_score(y_test_fold,predictions)
    model_folds_accuracy.append(fold_accuracy)
# Update the best classification model based on the accuracy
    if fold_accuracy > best_model_accuracy:
        best_model_accuracy = fold_accuracy
        best_model = model
        x_test_best_model=x_test_fold
        y_test_best_model= y_test_fold
# print best model accuracy and average accuracy
model_folds_accuracy = np.array(model_folds_accuracy)
print('Best model accuracy:', best_model_accuracy)
print('mean accuracy:', model_folds_accuracy.mean())
print('std accuracy:', model_folds_accuracy.std())
#Save the best model to disk
joblib.dump(best_model,'best_kFold_model.pkl')
# some time later...
# load the model from disk
loaded_model = joblib.load('best_kFold_model.pkl')
predictions_loaded_model = loaded_model.predict(x_test_best_model)
accuracy_loaded_model = accuracy_score(y_test_best_model,predictions_loaded_model)
print('Accuracy load model:', accuracy_loaded_model)