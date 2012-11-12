import os
import sys
import page_rank_update
import edge_computation

def predict_one_source(w, query_data, params):
	candidates = query_data["candidates"]
	(A, A_data) = compute_A(w, query_data["feature_stack"], query_data["edge_ij"], params["edge_strength_fun"], query_data["num_nodes"])
	Q = edge_computation.compute_Q(A, A_data, query_data["edge_ij"], query_data["s"], query_data["num_nodes"], params)
	p_0 = numpy.ones((query_data["num_nodes"], 1)) / (1.0 * query_data["num_nodes"])
	p = page_rank_update.update_p(p_0, Q, params)
	indices = numpy.argsort(-1 * p[candidates])
	p[candidates][indices] = numpy.sort(p[candidates])
	positives = candidates[indices[:K]]
	negatives = candidates[indices[K:]]
	return (positives, negatives)
