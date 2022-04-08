import pandas as pd
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[: ,1:2].values
y = dataset.iloc[:, 2:].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel = "rbf")
regressor.fit(X,y)

import matplotlib.pyplot as plt
plt.scatter(X, y , color="red")
plt.plot(X, regressor.predict(X), color="blue")
plt.title("SVR")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

import numpy as np
sc_X_val = sc_X.transform(np.array([[6.5]]))
scaled_y_pred = regressor.predict(sc_X_val)
y_pred = sc_y.inverse_transform(scaled_y_pred) 
print('The predicted salary of a person at 6.5 Level is ',y_pred)
