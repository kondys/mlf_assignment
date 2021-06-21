'''
Small test util to manually verify the core LR equation
'''

import numpy as np

weights = [-22.56148484,   0.8306227 ]
bias = 0.0021731246396342095

def sigmoid(x):                                 
    return 1 / (1 + np.exp(-x))

def predict(X):
    linear_model = np.dot(X, weights) + bias
    prediction = sigmoid(linear_model)
    result = [1 if i >= 0.5 else 0 for i in prediction]

    return np.array(result)

print(predict([[29,	689], [29,689], [19,268], [15,337], [33,667], [4,324], [17,554], [9,358]] )) 