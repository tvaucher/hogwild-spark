import numpy as np
from operator import add
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
import math

from utils import sign

class SVM:
    def __init__(self, learning_rate, lambda_reg, dim):
        self.__learning_rate = learning_rate
        self.__lambda_reg = lambda_reg
        self.__dim = dim
        self.__persistence = 15
        #self.__epsilon = 1e-4
        # initialize weights
        self.__w = np.zeros(dim)

    def fit(self, data, validation, max_iter, batch_size=100):
        reached_criterion = False
        early_stopping_window = []
        window_smallest = math.inf
        for i in range(max_iter):
            if(not reached_criterion) :
                grad, train_loss = self.step(data.sample(False, 0.01))
                self.__w += self.__learning_rate*grad.toarray().ravel()
                #train_accuracy = self.predict(data) #, train_accuracy : {train_accuracy:.2f},
                validation_loss = validation.map(
                    lambda x: self.loss(x[0], x[1])).reduce(add)
                validation_accuracy = self.predict(validation)
                print(f'''iter : {i:3d}, avg_train_loss : {train_loss:.4f} validation_loss : {validation_loss:.2f},\
                        validation_accuracy : {validation_accuracy}''')

                if(len(early_stopping_window) == self.__persistence) :
                    early_stopping_window = early_stopping_window[1:]
                    early_stopping_window.append(validation_loss)
                    if(min(early_stopping_window) > window_smallest):
                        reached_criterion = True
                        print('reached early stopping criterion')
                    window_smallest = min(early_stopping_window)
                else :
                    early_stopping_window.append(validation_loss)



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
        xw = x.dot(self.__w)[0]
        if self.__misclassification(xw, label):
            delta_w = self.__gradient(x, label)
        else:
            delta_w = self.__regularization_gradient(x)
        return delta_w, self.loss(x, label, xw=xw)

    def loss(self, x, label, xw=None):
        if xw is None:
            xw = x.dot(self.__w)[0]
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
