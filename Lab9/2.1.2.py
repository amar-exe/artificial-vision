# import the necessary packages
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# compute the classification confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, GaussianNB_predictions).ravel()
# compute the classification accuracy
GaussianNB_accuracy = accuracy_score(y_test, GaussianNB_predictions)
# compute the classification precision
GaussianNB_precision = precision_score(y_test, GaussianNB_predictions)
# compute the classification recall
GaussianNB_recall = recall_score(y_test, GaussianNB_predictions)
# compute the classification f1 score
GaussianNB_f1Score = f1_score(y_test, GaussianNB_predictions)

def accuracy_score(tp,tn,fp,fn):
    return ((tp+tn)/(tp+tn+fp+fn))

def precision_score(tp,fp):
    return (tp/(tp+fp))

def recall_score(tp,fn):
    return (tp/(tp+fn))

def f1_score(tp,fp,fn):
    return (tp/(tp+0.5*(fp+fn)))
