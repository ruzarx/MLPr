import numpy as np
import errors

class Linear_regression:

    '''
    Linear regression class, which provides calculation of coefficients for linear regression.

    It implements variation of several parameters:
        - learning_rate - coefficient of smoothing weights update value during gradient descent.
            The smaller the rate, the longer convergion will be, but the less chance of divergion.
        - epsilon - target calculation precision. When difference between adjucent iterations losses 
            becomes less than epsilon, evaluation stops.
        - max_rounds - maximum number of iterations before process termination.

    '''

    def __init__(self, learning_rate=0.001, epsilon=1e-8, max_rounds=10000):
        self.intercept = 0
        self.slope = 0
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.errors = []

    def fit(self, X, y):
        # Check X, y
        X = self.__data_check(X)
        y = self.__data_check(y)

        self.W = np.ones((X.shape[1] + 1, 1))
        train_X = self.__add_dimension(X)
        self.__learn(train_X, y)

    def __learn(self, X, y):
        prev_loss = np.inf
        loss = np.inf
        for i in range(self.max_rounds):
            grad_vector = self.__gradient(X, y)
            self.W = self.__weight_update(grad_vector)
            if loss == np.inf:
                loss = self.__loss(X, y)
                continue
            else:
                prev_loss = loss
                loss = self.__loss(X, y)
            self.errors.append(loss - prev_loss)
            if loss - prev_loss < self.epsilon:
                break
        return            

    def predict(self, x):
        # Check x, should be (X.shape[0], n) 
        test_X = self.__add_dimension(x)
        return self.__forward_pass(test_X)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    def __gradient(self, X, y):
        return 2 * (np.dot(X.T, np.dot(X, self.W) - y))

    def __weight_update(self, grad_vector):
        self.W -= self.learning_rate * grad_vector
        return self.W

    def __forward_pass(self, x):
        return np.dot(x, self.W)

    def __add_dimension(self, X):
        bigger_X = np.ones((X.shape[0], X.shape[1] + 1))
        bigger_X[:,1:] = X
        return bigger_X

    def __loss(self, X, y):
        # Different losses
        return np.sum(np.power((np.dot(X, self.W) - y), 2))

    def __data_check(self, X):
        if type(X) == list:
            X = np.array(X)
        if type(X) == np.ndarray:
            if len(X.shape) < 2:
                X = X.reshape((X.shape[0], 1))
        else:
            pass

        return X