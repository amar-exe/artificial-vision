
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# load the Breast Cancer Wisconsin (BCW) dataset
BCW_dataset = load_breast_cancer()
#Assign the feature vectors (data) and target to separate variables
x= BCW_dataset.data
y= BCW_dataset.target
#Splitting the data into training and testing sets
(x_train, x_test,y_train, y_test) = train_test_split(x,y,random_state=3,test_size=0.20)
