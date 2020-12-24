import numpy as np


def batch_linear_gradient_descent(x, y, learning_rate, iter_times):
    """
    x's shape: (num_samples, dimension)
    y's shape: (num_samples, 1)
    参数theta初始化为0
    返回经过iter_times轮学习率为learning_rate的梯度下降后的参数theta
    """
    dim = x.shape[1]
    theta = np.zeros((dim, 1))
    for i in range(iter_times):
        gradient = np.dot(x.transpose(), np.dot(x, theta) - y)
        theta -= learning_rate * gradient
    return theta


def stochastic_gradient_descent(x, y, learning_rate, iter_times):
    """
    x's shape: (num_samples, dimension)
    y's shape: (num_samples, 1)
    参数theta初始化为0
    返回经过iter_times轮学习率为learning_rate的随机梯度下降后的参数theta
    """
    num_samples, dim = x.shape
    theta = np.zeros((dim, 1))
    for i in range(iter_times):
        index = np.random.randint(0, num_samples)
        theta -= learning_rate * (np.dot(x[index], theta) - y[index]) * x[index].reshape(-1, 1)
    return theta


def mini_batch_linear_gradient_descent(x, y, learning_rate, batch_size, num_epochs):
    num_samples, dim = x.shape
    theta = np.zeros((dim, 1))
    indexes = np.arange(num_samples)
    np.random.shuffle(indexes)
    for i in range(num_epochs):
        for j in range((num_epochs + batch_size - 1) // batch_size):
            batch_indexes = indexes[j * batch_size: (j + 1) * batch_size]
            gradient = np.dot(x[batch_indexes].transpose(), np.dot(x[batch_size], theta) - y[batch_indexes])
            theta -= learning_rate * gradient
    return theta


a = np.random.randn(100, 10)
b = np.random.randn(100, 1)
lr = 0.1
times = 100

print(mini_batch_linear_gradient_descent(a, b, lr, 10, 50))


