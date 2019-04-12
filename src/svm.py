import numpy as np
from operator import add
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm

from utils import sign


class SVM:
    def __init__(self, learning_rate, lambda_reg, dim):
        self.__learning_rate = learning_rate
        self.__lambda_reg = lambda_reg
        self.__dim = dim
        # initialize weights
        self.__w = np.zeros(dim)

    def fit(self, data, max_iter, batch_size=100):
        for i in range(max_iter):
            print(
                f'accuracy before iter {i} : accuracy : {self.predict(data):.2f}')
            grad, train_loss = self.step(data.sample(False, 0.01))
            self.__w += self.__learning_rate*grad.toarray().ravel()
            print(f'iter : {i}, loss : {train_loss}')

    def step(self, data):
        '''
        Calculates the gradient and train loss.
        If the update flag is set to False, gradient is calculated but own weights will not be updated
        '''
        gradient, train_loss = data.map(lambda x: self.calculate_grad_loss(
            x[0], x[1])).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
        train_loss /= data.count()

        return gradient, train_loss

    def calculate_grad_loss(self, x, label):
        xw = x.dot(self.__w)
        if self.__misclassification(xw, label):
            delta_w = self.__gradient(x, label)
        else:
            delta_w = self.__regularization_gradient(x)
        return delta_w, self.loss(x, label, xw=xw)

    def loss(self, x, label, xw=None):
        if xw is None:
            xw = x.dot(self.__w)
        return max(1 - label * xw, 0) + self.__regularizer(x)

    def __regularizer(self, x):
        ''' Returns the regularization term '''
        w = self.__w
        return self.__lambda_reg * (w[x.indices]**2).sum()/x.nnz

    def __regularizer_g(self, x):
        '''Returns the gradient of the regularization term  '''
        w = self.__w
        return 2 * self.__lambda_reg * w[x.indices].sum()/x.nnz

    def __gradient(self, x, label):
        ''' Returns the gradient of the loss with respect to the weights '''
        grad = x.copy() * label
        grad.data -= self.__regularizer_g(x)
        return grad

    def __regularization_gradient(self, x):
        ''' Returns the gradient of the regularization term for each datapoint '''
        return csr_matrix((np.array([-self.__regularizer_g(x)]*x.nnz), x.indices, x.indptr), (1, self.__dim))

    def __misclassification(self, x_dot_w, label):
        ''' Returns true if x is misclassified. '''
        return x_dot_w * label < 1

    def predict(self, data):
        ''' Predict the labels of the input data '''
        return data.map(lambda x: sign(x[0].dot(self.__w)) == x[1]).reduce(add)/data.count()
