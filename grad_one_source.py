import os
import sys
import numpy
import scipy.sparse

#This is inefficient, but we only have to use it once!
def build_diff_generating_mat(positives, negatives, num_nodes):
	ij = numpy.zeros((2, 2 * len(positives) * len(negatives)))
	data = numpy.zeros(2 * len(positives) * len(negatives))
	t = 0
	for p in positives:
		for n in negatives:
			data[t] = 1
			ij[0, t] = t / 2
			ij[1, t] = n
			data[t + 1] = -1
			ij[0, t + 1] = t / 2
			ij[1, t + 1] = p
			t += 2

	diff_generating_mat = scipy.sparse.csr_matrix((data, ij), shape = (num_nodes, num_nodes))

	return diff_generating_mat

def sigmoid_h_loss_grad(diffs, margin):
	h = cost_one_source.sigmoid_h_loss(diffs, margin)
	return h * (1 - h) / margin

#-p is from a previous iteration
#-p_grad is from a previous iteration
def grad_one_source(s, p_warm_start, p_grad_warm_start, w, training_data, params):
	(A, A_data) = edge_computation.compute_A(w, training_data["feature_stack"], training_data["edge_ij"], params["edge_strength_fun"], training_data["num_nodes"])
	Q = edge_computation.compute_Q(A, A_data, training_data["edge_ij"], s, training_data["num_nodes"], params)
	Q_grad = []
	for k in range(training_data["num_features"]):
		df_dwk = edge_computation.compute_df_dwk(k, training_data["feature_stack"], A_data, training_data["edge_ij"], params["edge_strength_grad_fun"], training_data["num_nodes"], params)
		dQ_dwk = edge_computation.compute_dQ_dwk(k, df_dwk, A, params)
		Q_grad.append(dQ_dwk)

	p = page_rank_update.update_p(p_warm_start, Q, params)
	p_grad = partial_gradient_update.update_p_grad(p, p_grad_warm_start, Q, Q_grad, training_data["num_features"], params)

	positives = training_data["positives"]
	negatives = training_data["negatives"]
	candidates = list(set(postives + negatives))
	sum_p_candidates = numpy.sum(p[candidates])
	p_prime = p / sum_p_candidates #Okay, so this isn't full of zeros.  Sue me.

	diff_generating_mat = training_data["diff_generating_mat"]

	dpprime_dp_data = numpy.zeros(len(candidates) ** 2)
	dpprime_dp_ij = numpy.zeros((2, len(candidates) ** 2))
	t = 0
	for i in candidates:
		for j in candidates:
			dpprime_dp_ij[0, t] = i
			dpprime_dp_ij[1, t] = j
			
			if i == j:
				dpprime_dp_data[t] = (sum_p_candidates - p[j]) / (sum_p_candidates ** 2)
			else:
				dpprime_dp_data[t] = -1.0 * p[i] / (sum_p_candidates ** 2)

	dpprime_dp = scipy.sparse.csr_matrix((dpprime_dp_data, dpprime_dp_ij), shape = (training_data["num_nodes"], training_data["num_nodes"]))

	dpprime_dw = dpprime_dp.dot(p_grad) #Note: dpprime_dw is dense, even though it might have lots of zeros
	diff_generating_mat = training_data["diff_generating_mat"]
	diffs = diff_generating_mat.dot(p_prime)
	dpprime_dw_diffs = diff_generating_mat.dot(dpprime_dw) #This is (|L||D|) x num_features
	dh_ddiffs = params["h_grad_fun"](diffs, params["margin"])

	dh_dw = numpy.dot(dpprime_dw_diffs.T, dh_ddiffs)
	
	return (dh_dw, p, p_grad)
