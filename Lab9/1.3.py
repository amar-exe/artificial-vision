# import the necessary packages
from sklearn.datasets import load_breast_cancer
# load the Breast Cancer Wisconsin (BCW) dataset
BCW_dataset = load_breast_cancer()
# visualise some features examples
BCW_dataset.feature_names
BCW_dataset.data[[5,50]]
# visualise some target examples
BCW_dataset.target_names
BCW_dataset.target[[5,50]]