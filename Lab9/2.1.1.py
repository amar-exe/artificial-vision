from sklearn.naive_bayes import GaussianNB
# train the model
GaussianNB_model = GaussianNB()
GaussianNB_model.fit(x_train, y_train)
# make predictions using new data (testing set)
GaussianNB_predictions = GaussianNB_model.predict(x_test)