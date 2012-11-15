import numpy
import os
import sys
import scipy.optimize
import compute_cost
import compute_grad
import setup_params
import problem_setup
import grad_one_source
import multiprocessing

RETALL = False
CALLBACK_FUN = None

def train(training_data_list, p_warm_start_list, p_grad_warm_start_list, params, cost_fun, grad_fun, w0, pool = None):
	print(w0.shape)
	w_opt = scipy.optimize.fmin_bfgs(cost_fun, w0, fprime = grad_fun, args = (training_data_list, p_warm_start_list, p_grad_warm_start_list, params, pool), maxiter=params["maxiter"], retall = RETALL, callback = CALLBACK_FUN, gtol = 1e-1)
	print(w_opt)

	return w_opt

#def f(w, training_data, params):
#	print("Training data:")
#	print(training_data)
#	print("params:")
#	print(params)
#	return numpy.sum(numpy.square(w))
#
#def fgrad(w, training_data, params):
#	print("Training data:")
#	print(training_data)
#	print("params:")
#	print(params)
#	return 2 * w


if __name__ == "__main__":
	numpy.random.seed(int(sys.argv[1]))
	
	params = setup_params.setup_params()
	pool = multiprocessing.Pool(8)
	
	adjlist_file_name = "../synthetic_data.ajdlist0"
	feature_file_name = "../synthetic_data.feature0"
	
	num_graphs = 1
	
	training_data_list = [{}] * num_graphs
	for i in range(num_graphs):
		training_data_list[i]["num_features"] = 2
		training_data_list[i]["source"] = 0
		training_data_list[i]["positives"] = range(200)
		training_data_list[i]["negatives"] = range(9800, 10000)
		training_data_list[i]["num_nodes"] = 10000
		(training_data_list[i]["edge_ij"], training_data_list[i]["feature_stack"]) = problem_setup.get_edge_ij_and_feature_stack(adjlist_file_name, feature_file_name, training_data_list[i]["num_features"])
		training_data_list[i]["diff_generating_mat"] = grad_one_source.build_diff_generating_mat(range(200), range(200, 9800), 10000)
	
	w0 = numpy.random.randn(training_data_list[0]["num_features"], 1)
	print(w0.shape)
	
	p_warm_start_list = [numpy.ones((10000, 1)) / 10000.0] * num_graphs
	p_grad_warm_start_list = [numpy.zeros((10000, 2))] * num_graphs
	
	#w_1 = numpy.copy(w0)
	#w_1[0] -= 1e-5
	#w_2 = numpy.copy(w0)
	#w_2[0] += 1e-5
	#
	#cost_1 = compute_cost.compute_cost(w_1, training_data_list, p_warm_start_list, p_grad_warm_start_list, params, pool)
	#grad_1 = compute_grad.compute_grad(w_1, training_data_list, p_warm_start_list, p_grad_warm_start_list, params, pool)
	#
	#cost_2 = compute_cost.compute_cost(w_2, training_data_list, p_warm_start_list, p_grad_warm_start_list, params, pool)
	#grad_2 = compute_grad.compute_grad(w_2, training_data_list, p_warm_start_list, p_grad_warm_start_list, params, pool)
	#
	#print(cost_1)
	#print(grad_1)
	#print(cost_2)
	#print(grad_2)
	#print(cost_2 - cost_1)
	
	w_opt = train(training_data_list, p_warm_start_list, p_grad_warm_start_list, params, compute_cost.compute_cost, compute_grad.compute_grad, w0, pool = pool)
	
	numpy.save("blah_model.npy", w_opt)
