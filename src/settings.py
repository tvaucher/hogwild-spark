import os

path = '..' if os.environ.get('LOCAL', False) else '/data'
logpath = '../logs' if os.environ.get('LOCAL', False) else '/data/logs'

dim = 47238
learning_rate = 3
lambda_reg = 1e-4

validation_frac = 0.1
