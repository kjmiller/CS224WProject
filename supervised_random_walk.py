import numpy
import os
import sys
import scipy.optimize

RETALL = False
CALLBACK_FUN = None

def train(training_data, params, cost_fun, grad_fun, w0):
	w_opt = scipy.optimize.fmin_bfgs(cost_fun, w0, fprime = grad_fun, args = (training_data, params), epsilon=params["epsilon"], maxiter=params["maxiter"], retall = RETALL, callback = CALLBACK_FUN)
	print(w_opt)
	return w_opt

def f(w, training_data, params):
	print("Training data:")
	print(training_data)
	print("params:")
	print(params)
	return numpy.sum(numpy.square(w))

def fgrad(w, training_data, params):
	print("Training data:")
	print(training_data)
	print("params:")
	print(params)
	return 2 * w

w0 = 1.0 * numpy.ones((3, 3))
params = {}
params["epsilon"] = 1e-12
params["maxiter"] = 1000
w_opt = train(None, params, f, fgrad, w0)
