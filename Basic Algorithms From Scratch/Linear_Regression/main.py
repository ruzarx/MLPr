import linear_regressor
import numpy as np

X = np.array([3, 4, 15, 16, 17, 8, 9, 100])
X = X.reshape((X.shape[0], 1))
y = np.array([4, 4, 6, 5, 7, 9, 8, 10])
y = y.reshape((y.shape[0], 1))

lr = linear_regressor.Linear_regression(learning_rate=0.0001)
lr.fit_predict(X, y)
print(lr.W)