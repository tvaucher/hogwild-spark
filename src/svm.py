import numpy as np
from operator import add
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
import math


class SVM:
    def __init__(self, learning_rate, lambda_reg, batch_frac, dim):
        self.__learning_rate = learning_rate
        self.__lambda_reg = lambda_reg
        self.__batch_frac = batch_frac
        self.__dim = dim
        self.__persistence = 15
        self.__w = np.zeros(dim)

    def fit(self, data, validation, max_iter):
        reached_criterion = False
        early_stopping_window = []
        window_smallest = math.inf
        log = []
        for i in range(max_iter):
            if not reached_criterion:
                # Compute gradient and train loss
                grad, train_loss = self.step(
                    data.sample(False, self.__batch_frac))
                self.__w -= self.__learning_rate * (grad.toarray().ravel() + self.l2_reg_grad())
                # Compute validation loss and accuracy
                validation_loss = self.loss(validation)
                validation_accuracy = self.predict(validation)

                # Logging
                log_iter = {'iter': i, 'avg_train_loss': train_loss,
                            'validation_loss': validation_loss, 'validation_accuracy': validation_accuracy}
                log.append(log_iter)

                # Early stopping criterion
                if(len(early_stopping_window) == self.__persistence):
                    early_stopping_window = early_stopping_window[1:]
                    early_stopping_window.append(validation_loss)
                    if(min(early_stopping_window) > window_smallest):
                        reached_criterion = True
                        log.append({'early_stop': True})
                        break
                    window_smallest = min(early_stopping_window)
                else:
                    early_stopping_window.append(validation_loss)
        return log

    def step(self, data):
        '''
        Calculates the gradient and train loss.
        If the update flag is set to False, gradient is calculated but own weights will not be updated
        '''
        gradient, train_loss = data.map(lambda x: self.calculate_grad_loss(
            x[0], x[1])).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
        gradient /= data.count()

        return gradient, train_loss + self.l2_reg()

    def calculate_grad_loss(self, x, label):
        xw = x.dot(self.__w)[0]
        if self.misclassification(xw, label):
            return self.gradient(x, label), self.loss_point(x, label, xw=xw)
        else:
            return 0, 0

    def loss_point(self, x, label, xw=None):
        if xw is None:
            xw = x.dot(self.__w)[0]
        return max(1 - label * xw, 0)

    def loss(self, data):
        return data.map(lambda x: self.loss_point(x[0], x[1])).reduce(add) + self.l2_reg()

    def l2_reg(self):
        ''' Returns the regularization term '''
        w = self.__w
        return self.__lambda_reg * (w ** 2).sum()

    def l2_reg_grad(self):
        '''Returns the gradient of the regularization term  '''
        w = self.__w
        return 2 * self.__lambda_reg * w

    def gradient(self, x, label):
        ''' Returns the gradient of the loss with respect to the weights '''
        return -x*label

    def misclassification(self, x_dot_w, label):
        ''' Returns true if x is misclassified. '''
        return x_dot_w * label < 1

    def predict(self, data):
        ''' Predict the labels of the input data '''
        sign = lambda x : 1 if x > 0 else -1 if x < 0 else 0
        return data.map(lambda x: sign(x[0].dot(self.__w)) == x[1]).reduce(add)/data.count()
