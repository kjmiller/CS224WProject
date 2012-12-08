import os
import sys
import setup_params
import problem_setup
import numpy.random
import numpy
import networkx
import predict_one_source
import copy
import supervised_random_walk
import compute_cost
import compute_grad
import grad_one_source
import multiprocessing

pool = multiprocessing.Pool(1)

#We get gt's by taking training candidates, making the upper 500 of them be positive, and making the rest be negative
#To train, we randomly sample some proportion of the gt positives and negatives.  We still look at ALL the candidates when making predictions.

#train_test_file_name = "tiny_train_test_split.txt"
numpy.random.seed(int(sys.argv[1]))

sigma_squared = float(sys.argv[2])
#num_graphs = 50
num_nodes = 2000
params = setup_params.setup_params()

training_data_list = []

w_gt = numpy.array([[1.0], [-1.0]])

i = 0
indices = []
training_file = open("synthetic_trainers_mini.txt", "r")
for line in training_file:
	index = int(line.rstrip("\n"))
	indices.append(index)
	(edge_ij, feature_stack, G) = problem_setup.get_edge_ij_and_feature_stack("../synthetic_data/synthetic_data.ajdlist%d"%(index), "../synthetic_data/synthetic_data.antisymmetric_feature%d"%(index), 2, num_nodes = num_nodes)
	source = index % 3
	candidates = list(set(G.nodes()) - set(G.neighbors(source)) - set([source]))
	training_data_list.append({})
	training_data_list[i]["num_features"] = 2
	training_data_list[i]["source"] = source
	training_data_list[i]["candidates"] = candidates
	training_data_list[i]["edge_ij"] = edge_ij
	training_data_list[i]["feature_stack"] = feature_stack
	training_data_list[i]["num_nodes"] = num_nodes
	(training_data_list[i]["training_positives"], training_data_list[i]["training_negatives"]) = predict_one_source.predict_one_source(w_gt, training_data_list[i], params)
	problem_setup.write_spn_list("../synthetic_data/synthetic_data.spn%d"%(index), [(source, training_data_list[i]["training_positives"], training_data_list[i]["training_negatives"])])

	i += 1
#x = range(num_graphs)

training_file.close()

num_graphs = len(training_data_list)

#We will choose the bottom half for training and the top half for testing
#numpy.random.shuffle(x)
#train_test_file = open(train_test_file_name, "w")
#for i in range(len(x)):
#	train_test_file.write("%d\n"%(x[i]))

#train_test_file.close()

#training_data_list = map(lambda i: training_data_list[i], x[:(num_graphs)])
#testing_data_list = map(lambda i: training_data_list[i], x[(num_graphs / 2):])

for training_data in training_data_list:
	rpos = numpy.random.rand(len(training_data["training_positives"]))
	rneg = numpy.random.rand(len(training_data["training_negatives"]))
	training_data["positives"] = []
	for i in range(len(rpos)):
		if rpos[i] <= params["prop_training_gts"]:
			training_data["positives"].append(training_data["training_positives"][i])

	training_data["negatives"] = []
	for i in range(len(rneg)):
		if rneg[i] <= params["prop_training_gts"]:
			training_data["negatives"].append(training_data["training_negatives"][i])
	
	training_data["diff_generating_mat"] = grad_one_source.build_diff_generating_mat(training_data["positives"], training_data["negatives"], training_data["num_nodes"])

for i in range(num_graphs):
	(garbage_a, training_data_list[i]["feature_stack"], garbage_b) = (edge_ij, feature_stack, G) = problem_setup.get_edge_ij_and_feature_stack("../synthetic_data/synthetic_data.ajdlist%d"%(indices[i]), "../synthetic_data/synthetic_data.antisymmetric_feature%d_noisy%f"%(indices[i], sigma_squared), 2, num_nodes = num_nodes)

p_warm_start_list = [numpy.ones((num_nodes, 1)) / (1.0 * num_nodes)] * len(training_data_list)
p_grad_warm_start_list = [numpy.zeros((num_nodes, training_data_list[0]["num_features"]))] * len(training_data_list)

w0 = numpy.random.randn(2, 1) #numpy.array([[1.0], [-1.0]])

w_opt = supervised_random_walk.train(training_data_list, p_warm_start_list, p_grad_warm_start_list, params, compute_cost.compute_cost, compute_grad.compute_grad, w0, pool = pool)

numpy.save("model_synthetic_backprop_mini_noisy%f.npy"%(sigma_squared), w_opt)
