import os
import sys
import page_rank_update
import edge_computation
import numpy

def predict_one_source(w, query_data, params):
	K = params["K"]
	candidates = query_data["candidates"]
	(A, A_data) = edge_computation.compute_A(w, query_data["feature_stack"], query_data["edge_ij"], params["edge_strength_fun"], query_data["num_nodes"])
	Q = edge_computation.compute_Q(A, A_data, query_data["edge_ij"], query_data["source"], query_data["num_nodes"], params)
	p_0 = numpy.ones((query_data["num_nodes"], 1)) / (1.0 * query_data["num_nodes"])
	p = page_rank_update.update_p(p_0, Q, params)
	indices = numpy.argsort(-1.0 * p[candidates], axis = 0)
	positives = []
	negatives = []
		
	for k in range(indices.shape[0]):
		if k < K:
			positives.append(candidates[indices[k, 0]])
		else:
			negatives.append(candidates[indices[k, 0]])
	
	#print("min(p+) - max(p-) = %f"%(numpy.amin(p[positives]) - numpy.amax(p[negatives])))
	#print("max(p+) - min(p-) = %f"%(numpy.amax(p[positives]) - numpy.amin(p[negatives])))
	return (positives, negatives)
