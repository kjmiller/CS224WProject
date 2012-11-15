import os
import sys
import problem_setup
import setup_params
import predict_one_source
import numpy

params = setup_params.setup_params()

num_nodes = 2000
testing_file = open("synthetic_testers_mini.txt", "r")
testing_data_list = []

w_gt = numpy.array([[1.0], [-1.0]])

aucs = []
indices = []
i=0
for line in testing_file:
	testing_data_list.append({})
	index = int(line.rstrip("\n"))
	indices.append(index)
	(testing_data_list[i]["edge_ij"], testing_data_list[i]["feature_stack"], G) = problem_setup.get_edge_ij_and_feature_stack("../synthetic_data/synthetic_data.ajdlist%d"%(index), "../synthetic_data/synthetic_data.antisymmetric_feature%d"%(index), 2, num_nodes = num_nodes)
	source = index % 3
	candidates = list(set(G.nodes()) - set(G.neighbors(source)) - set([source]))
	testing_data_list[i]["num_features"] = 2
	testing_data_list[i]["source"] = source
	testing_data_list[i]["candidates"] = candidates
	testing_data_list[i]["num_nodes"] = num_nodes
	(testing_data_list[i]["positives"], testing_data_list[i]["negatives"]) = predict_one_source.predict_one_source(w_gt, testing_data_list[i], params)
	problem_setup.write_spn_list("../synthetic_data/synthetic_data.spn%d"%(index), [(source, testing_data_list[i]["positives"], testing_data_list[i]["negatives"])])
	
	i += 1


for sigma_squared in [0.0, 0.5, 1.0, 1.5, 2.0]:
	num_true_positives = [0.0] * num_nodes
	num_true_negatives = [0.0] * num_nodes
	num_false_positives = [0.0] * num_nodes
	num_false_negatives = [0.0] * num_nodes
	print(str(sigma_squared))
	#w_opt = numpy.load("model_synthetic_mini_noisy%f.npy"%(sigma_squared))
	w_opt = numpy.array([[1.0], [-1.0]])
	for i in range(len(indices)):
		(sdfsfs, testing_data_list[i]["feature_stack"], sdfs) = problem_setup.get_edge_ij_and_feature_stack("../synthetic_data/synthetic_data.ajdlist%d"%(indices[i]), "../synthetic_data/synthetic_data.antisymmetric_feature%d_noisy%f"%(indices[i], sigma_squared), 2, num_nodes = num_nodes)
		for K in range(num_nodes / 200):
			print(str(K * 200))
			params["K"] = min(len(testing_data_list[i]["candidates"]), K * 200)
			(predicted_positives, predicted_negatives) = predict_one_source.predict_one_source(w_opt, testing_data_list[i], params)
			for p in predicted_positives:
				if p in testing_data_list[i]["positives"]:
					num_true_positives[K] += 1.0
				elif p in testing_data_list[i]["negatives"]:
					num_false_positives[K] += 1.0
				else:
					assert(0)

			for n in predicted_negatives:
				if n in testing_data_list[i]["positives"]:
					num_false_negatives[K] += 1.0
				elif n in testing_data_list[i]["negatives"]:
					num_true_negatives[K] += 1.0
				else:
					assert(0)


	tp = []
	fp = []
	for K in range(num_nodes / 200):
		tp.append(num_true_positives[K] / (num_true_positives[K] + num_false_negatives[K]))
		fp.append(num_false_positives[K] / (num_false_positives[K] + num_true_negatives[K]))

	auc = 0.0
	for i in range(len(tp) - 1):
		auc += (fp[i + 1] - fp[i]) * (tp[i] + tp[i + 1]) / 2.0

	aucs.append(auc)

print(aucs)
print(sigma_squared)
