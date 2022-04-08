
import pandas as pd
dataset = pd.read_csv("iphone_purchase_records.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

from sklearn.preprocessing import LabelEncoder
labelEncoder_gender =  LabelEncoder()
X[:,0] = labelEncoder_gender.fit_transform(X[:,0])

import numpy as np
X = np.vstack(X[:, :]).astype(np.float)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=97)

from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = "linear", random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred) 
print(cm)
accuracy = metrics.accuracy_score(y_test, y_pred) 
print("Accuracy score:",accuracy)
precision = metrics.precision_score(y_test, y_pred) 
print("Precision score:",precision)
recall = metrics.recall_score(y_test, y_pred) 
print("Recall score:",recall)

#kernel SVM

# classifier = SVC( kernel="rbf", random_state=97)
# classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)

# cm = metrics.confusion_matrix(y_test, y_pred) 
# print(cm)
# accuracy = metrics.accuracy_score(y_test, y_pred) 
# print("Accuracy score:",accuracy)
# precision = metrics.precision_score(y_test, y_pred) 
# print("Precision score:",precision)
# recall = metrics.recall_score(y_test, y_pred) 
# print("Recall score:",recall)

