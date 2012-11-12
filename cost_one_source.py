import numpy
import page_rank_update
import edge_computation

def sigmoid_h_loss(diffs, margin):
	return 1 / (1 + numpy.exp(-1.0 * diffs / margin))

def cost_one_source(w, s, training_data, p_warm_start, params):
	(A, A_data) = compute_A(w, training_data["feature_stack"], training_data["edge_ij"], params["edge_strength_fun"], training_data["num_nodes"])
	Q = compute_Q(A, A_data, training_data["edge_ij"], s, training_data["num_nodes"], params)
	p = page_rank_update.update_p(p_warm_start, Q, params)
	loss_fun = params["loss_fun"]
	margin = params["margin"]
	positives = training_data["positives"]
	negatives = training_data["negatives"]
	candidates = list(set(postives + negatives))
	p_prime = p / numpy.sum(p[candidates])

	diff_generating_mat = training_data["diff_generating_mat"]
	diffs = diff_generating_mat.dot(p_prime)

	loss = numpy.sum(loss_fun(diffs, margin))

	return (loss, p)
