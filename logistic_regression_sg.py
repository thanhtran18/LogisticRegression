# # import numpy as np
# # from scipy.io import *
# # import matplotlib.pyplot as plt
# # from utils import *
# # import time
# #
# # # Maximum number of iterations. Continue until this limit, or when erro change is below tol.
# # max_iter = 500
# # tol = 0.00001
# #
# # # Step size for gradient descent
# # eta = 0.003
# # # Get X1,X2
# # data = loadmat('data.mat')
# # # print(data)
# #
# # X1, X2 = data['X1'], data['X2']
# #
# # # Data matrix,m with column of ones at end.
# # X = np.vstack((X1, X2))
# # X = np.hstack((X, np.ones((X.shape[0], 1))))
# # np.random.permutation(X)
# # # Target values, 0 for class 1 (datapoints X1), 1 for class 2 (datapoints X2)
# # t = np.vstack((np.zeros((X1.shape[0], 1)), np.ones((X2.shape[0], 1))))
# #
# # # Initialize w.
# # w = np.array([1, 0, 0])
# #
# # # Error values over all iterations
# # e_all = np.array([])
# #
# # # Set up the slope-intercept figure
# # plt.figure(2)
# # plt.rcParams['font.size'] = 20
# # plt.title('Separator in slope-intercept space')
# # plt.xlabel('slope')
# # plt.ylabel('intercept')
# # plt.axis([-5, 5, -10, 0])
# #
# # for iter in range(max_iter):
# #     # w_old = 0
# #     # e = 0
# #     # y = sigmoid(w.T @ X.T).T
# #     # # # e is the rror, negative log-likelihood
# #     # e = -np.sum(t * np.log(y) + (1 - t) * np.log(1 - y))
# #     # #
# #     # # # Add this error to the end of error vector
# #     # e_all = np.append(e_all, e)
# #     # w_old = w
# #     for i in range(X.shape[0]):
# #         # Compute output using current w on all data X.
# #         y = sigmoid(w.T @ X[i, :].T).T
# #
# #         # # e is the rror, negative log-likelihood
# #         # e = -np.sum(t * np.log(y) + (1 - t[i]) * np.log(1 - y))
# #         #
# #         # # Add this error to the end of error vector
# #         # e_all = np.append(e_all, e)
# #
# #         # Gradient of the error, using Eqn 4.91
# #         grad_e = ((y - t[i]) * X[i, :])  # 1-by-3
# #         # grad_e = np.sum((y - t[i]) * X[i, :], 0, keepdims=True)
# #
# #         # Update w, *subtracking* a step in the error derivative since we are minimizing
# #         w_old = w
# #         w = w - eta * grad_e
# #         # y = sigmoid(w.T @ X.T).T
# #
# #     y = sigmoid(w.T @ X.T).T
# #     # e is the rror, negative log-likelihood
# #     e = -np.sum(t * np.log(y) + (1 - t) * np.log(1 - y))
# #     # e = -np.mean(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))
# #     # Add this error to the end of error vector
# #     e_all = np.append(e_all, e)
# #     # if 1:
# #     #     # Plot current separator and data
# #     #     plt.figure(1)
# #     #     plt.clf()
# #     #     plt.rcParams['font.size'] = 20
# #     #     plt.plot(X1[:, 0], X1[:, 1], 'g.')
# #     #     plt.plot(X2[:, 0], X2[:, 1], 'b.')
# #     #     drawSep(plt, w)
# #     #     plt.title('Separator in data space')
# #     #     plt.axis([-5, 15, -10, 10])
# #     #     plt.draw()
# #     #     plt.pause(1e-17)
# #
# #     # Add next step of separator in m-b space
# #     plt.figure(2)
# #     plotMB(plt, w, w_old)
# #     plt.draw()
# #     # plt.pause(1e-17)
# #
# # # Print some information
# #     print('iter %d, negative log-likelihood %.4f, w=%s' % (iter, e, np.array2string(w.T)))
# #
# #     # Stop iterating if error does not change more than tol
# #     if iter > 0:
# #         if abs(e - e_all[iter - 1]) < tol:
# #             break
# #         # plt.figure(2)
# #         # plotMB(plt, w, w_old)
# #         # plt.draw()
# #
# #
# # # Plot error over iterations
# # plt.figure(3)
# # plt.rcParams['font.size'] = 20
# # plt.plot(e_all, 'b-')
# # plt.xlabel('Iteration')
# # plt.ylabel('neg. log likelihood')
# # plt.title('Minimization using Stochastics gradient descent')
# #
# # plt.show()
#

"""********************************************"""

import numpy as np
from scipy.io import *
import matplotlib.pyplot as plt
from utils import *
import time

# Maximum number of iterations. Continue until this limit, or when erro change is below tol.
max_iter = 500
tol = 0.000001

# Step size for gradient descent
eta = 0.003
# Get X1,X2
data = loadmat('data.mat')
# print(data)
# np.random.permutation(data['X1'])
# np.random.permutation(data['X2'])
X1, X2 = data['X1'], data['X2']

# Data matrix,m with column of ones at end.
X = np.vstack((X1, X2))
X = np.hstack((X, np.ones((X.shape[0], 1))))
np.random.permutation(X)
# Target values, 0 for class 1 (datapoints X1), 1 for class 2 (datapoints X2)
t = np.vstack((np.zeros((X1.shape[0], 1)), np.ones((X2.shape[0], 1))))

# Initialize w.
# w = np.array([1, 0, 0]).reshape(3 ,1)
w = np.array([1, 0, 0])

# Error values over all iterations
e_all = np.array([])

# Set up the slope-intercept figure
plt.figure(2)
plt.rcParams['font.size'] = 20
plt.title('Separator in slope-intercept space')
plt.xlabel('slope')
plt.ylabel('intercept')
plt.axis([-5, 5, -10, 0])

for iter in range(max_iter):
    # w_old = 0
    # e = 0
    # y = sigmoid(w.T @ X.T).T
    # # # e is the rror, negative log-likelihood
    # e = -np.sum(t * np.log(y) + (1 - t) * np.log(1 - y))
    # #
    # # # Add this error to the end of error vector
    # e_all = np.append(e_all, e)
    # w_old = w
    for i in range(X.shape[0]):
        # Compute output using current w on all data X.
        y = sigmoid(w.T @ X[i, :])

        # # e is the rror, negative log-likelihood
        # e = -np.sum(t * np.log(y) + (1 - t[i]) * np.log(1 - y))
        #
        # # Add this error to the end of error vector
        # e_all = np.append(e_all, e)

        # Gradient of the error, using Eqn 4.91
        grad_e = ((y - t[i]) * X[i, :])  # 1-by-3
        # grad_e = np.sum((y - t[i]) * X[i, :], 0, keepdims=True)

        # Update w, *subtracking* a step in the error derivative since we are minimizing
        w_old = w
        w = w - eta * grad_e.T
        # y = sigmoid(w.T @ X.T).T

    # y = sigmoid(w.T @ X.T).T
    # e is the rror, negative log-likelihood
    e = -np.sum(t * np.log(y) + (1 - t) * np.log(1 - y))
    # e = -np.mean(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))
    # Add this error to the end of error vector
    e_all = np.append(e_all, e)
    # if 1:
    #     # Plot current separator and data
    #     plt.figure(1)
    #     plt.clf()
    #     plt.rcParams['font.size'] = 20
    #     plt.plot(X1[:, 0], X1[:, 1], 'g.')
    #     plt.plot(X2[:, 0], X2[:, 1], 'b.')
    #     drawSep(plt, w)
    #     plt.title('Separator in data space')
    #     plt.axis([-5, 15, -10, 10])
    #     plt.draw()
    #     plt.pause(1e-17)

    # Add next step of separator in m-b space
    plt.figure(2)
    plotMB(plt, w, w_old)
    plt.draw()
    # plt.pause(1e-17)

# Print some information
    print('iter %d, negative log-likelihood %.4f, w=%s' % (iter, e, np.array2string(w.T)))

    # Stop iterating if error does not change more than tol
    if iter > 0:
        if abs(e - e_all[iter - 1]) < tol:
            break
        # plt.figure(2)
        # plotMB(plt, w, w_old)
        # plt.draw()


# Plot error over iterations
plt.figure(3)
plt.rcParams['font.size'] = 20
plt.plot(e_all, 'b-')
plt.xlabel('Iteration')
plt.ylabel('neg. log likelihood')
plt.title('Minimization using Stochastics gradient descent')

plt.show()


"""****************** 3 LOOPS"""
# import numpy as np
# from scipy.io import *
# import matplotlib.pyplot as plt
# from utils import *
# import time
#
# # Maximum number of iterations. Continue until this limit, or when erro change is below tol.
# max_iter = 500
# tol = 0.000001
#
# # Step size for gradient descent
# eta = 0.003
# # Get X1,X2
# data = loadmat('data.mat')
# # print(data)
# # np.random.permutation(data['X1'])
# # np.random.permutation(data['X2'])
# X1, X2 = data['X1'], data['X2']
#
# # Data matrix,m with column of ones at end.
# X = np.vstack((X1, X2))
# X = np.hstack((X, np.ones((X.shape[0], 1))))
# np.random.permutation(X)
# # Target values, 0 for class 1 (datapoints X1), 1 for class 2 (datapoints X2)
# t = np.vstack((np.zeros((X1.shape[0], 1)), np.ones((X2.shape[0], 1))))
#
# # Initialize w.
# # w = np.array([1, 0, 0]).reshape(3 ,1)
# w = np.array([1, 0, 0])
#
# # Error values over all iterations
# e_all = np.array([])
#
# # Set up the slope-intercept figure
# plt.figure(2)
# plt.rcParams['font.size'] = 20
# plt.title('Separator in slope-intercept space')
# plt.xlabel('slope')
# plt.ylabel('intercept')
# plt.axis([-5, 5, -10, 0])
#
# for iter in range(max_iter):
#     # w_old = 0
#     # e = 0
#     # y = sigmoid(w.T @ X.T).T
#     # # # e is the rror, negative log-likelihood
#     # e = -np.sum(t * np.log(y) + (1 - t) * np.log(1 - y))
#     # #
#     # # # Add this error to the end of error vector
#     # e_all = np.append(e_all, e)
#     # w_old = w
#     for i in range(X.shape[0]):
#         for j in range(3):
#             # Compute output using current w on all data X.
#             y = sigmoid(w.T @ X[i, :])
#
#             # # e is the rror, negative log-likelihood
#             # e = -np.sum(t * np.log(y) + (1 - t[i]) * np.log(1 - y))
#             #
#             # # Add this error to the end of error vector
#             # e_all = np.append(e_all, e)
#
#             # Gradient of the error, using Eqn 4.91
#             # grad_e = ((y - t[i]) * X[i, :])  # 1-by-3
#             grad_e = np.sum((y - t[i]) * X[i, j], 0, keepdims=True)
#
#             # Update w, *subtracking* a step in the error derivative since we are minimizing
#             w_old = w
#             w = w - eta * grad_e.T
#             # y = sigmoid(w.T @ X.T).T
#
#         # y = sigmoid(w.T @ X.T).T
#         # e is the rror, negative log-likelihood
#         e = -np.sum(t * np.log(y) + (1 - t) * np.log(1 - y))
#         # e = -np.mean(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))
#         # Add this error to the end of error vector
#         e_all = np.append(e_all, e)
#         # if 1:
#         #     # Plot current separator and data
#         #     plt.figure(1)
#         #     plt.clf()
#         #     plt.rcParams['font.size'] = 20
#         #     plt.plot(X1[:, 0], X1[:, 1], 'g.')
#         #     plt.plot(X2[:, 0], X2[:, 1], 'b.')
#         #     drawSep(plt, w)
#         #     plt.title('Separator in data space')
#         #     plt.axis([-5, 15, -10, 10])
#         #     plt.draw()
#         #     plt.pause(1e-17)
#
#         # Add next step of separator in m-b space
#         plt.figure(2)
#         plotMB(plt, w, w_old)
#         plt.draw()
#         # plt.pause(1e-17)
#
#     # Print some information
#         print('iter %d, negative log-likelihood %.4f, w=%s' % (iter, e, np.array2string(w.T)))
#
#         # Stop iterating if error does not change more than tol
#         if iter > 0:
#             if abs(e - e_all[iter - 1]) < tol:
#                 break
#             # plt.figure(2)
#             # plotMB(plt, w, w_old)
#             # plt.draw()
#
#
# # Plot error over iterations
# plt.figure(3)
# plt.rcParams['font.size'] = 20
# plt.plot(e_all, 'b-')
# plt.xlabel('Iteration')
# plt.ylabel('neg. log likelihood')
# plt.title('Minimization using Stochastics gradient descent')
#
# plt.show()