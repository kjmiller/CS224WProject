import os
import sys
import numpy
import grad_one_source
import copy

def call_compute_grad_one_source(arg_tuple):
	return grad_one_source(arg_tuple[0], arg_tuple[1], arg_tuple[2], arg_tuple[3], arg_tuple[4], arg_tuple[5])

def create_arg_tuples(w, training_data_list, p_warm_start_list, p_grad_warm_start_list, params):
	arg_tuples = []
	for i in range(len(training_data_list)):
		arg_tuples.append((training_data_list[i]["sources"], p_warm_start_list[i], p_grad_warm_start_list[i], w.deepcopy(), training_data_list[i], copy.deepcopy(params)))
	
	return arg_tuples

def compute_grad(w, training_data_list, p_warm_start_list, p_grad_warm_start_list, params, pool = None):
	arg_tuples = create_arg_tuples(w, training_data_list, p_warm_start_list, p_grad_warm_start_list, params)
	if pool == None:
		result_tuples = map(call_compute_grad_one_source, arg_tuples)
	else:
		result_tuples = pool.map(call_compute_grad_one_source, arg_tuples)

	w_grad = numpy.zeros(w.shape)
	i = 0
	for result_tuple in result_tuples:
		w_grad += result_tuple[0]
		p_warm_start_list[i] = result_tuple[1]
		p_grad_warm_start_list[i] = result_tuple[2]
		i = i + 1

	w_grad *= params["lambda"]
	w_grad += 2 * w

	return w_grad
