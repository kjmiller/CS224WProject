import os
import sys
import numpy
import cost_one_source
import multiprocessing
import copy

#training_data_list = [{"source" : 0, "positives" : [1,2], "negatives" : [3, 4], "feature_stack" : sdfsdf, "edge_ij" : sdfsdf, ...}, {"source" : 1,..."edge_ij" : sdfsdf...},...]

def call_compute_cost_one_source(arg_tuple):
	return cost_one_source.cost_one_source(arg_tuple[0], arg_tuple[1], arg_tuple[2], arg_tuple[3])

def create_arg_tuples(w, training_data_list, p_warm_start_list, params):
	arg_tuples = []
	for i in range(len(training_data_list)):
		arg_tuples.append((w.copy(), training_data_list[i], p_warm_start_list[i], copy.deepcopy(params)))
	return arg_tuples

def compute_cost(w, training_data_list, p_warm_start_list, p_grad_warm_start_list, params, pool = None):
	print("==========")
	print("HERE IS W:")
	print(w)
	print("==========")
	w = numpy.reshape(w, (len(w), 1))
	arg_tuples = create_arg_tuples(w, training_data_list, p_warm_start_list, params)
	if pool == None:
		result_tuples = map(call_compute_cost_one_source, arg_tuples)
	else:
		result_tuples = pool.map(call_compute_cost_one_source, arg_tuples)

	total_loss = 0.0
	i = 0
	for result_tuple in result_tuples:
		total_loss += result_tuple[0]
		p_warm_start_list[i] = result_tuple[1]
		i += 1

	total_loss *= params["lambda"]
	total_loss += numpy.sum(numpy.square(w))

	print("cost = " + str(total_loss))

	print("mean positive probability = %f"%(numpy.mean(p_warm_start_list[0][training_data_list[0]["positives"]])))
	print("mean negative probability = %f"%(numpy.mean(p_warm_start_list[0][training_data_list[0]["negatives"]])))

	return total_loss
