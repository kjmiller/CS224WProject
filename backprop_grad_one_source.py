import numpy
import page_rank_update
import grad_one_source
import edge_computation

def compute_top_layer_delta(dpprime_dp, diff_generating_mat, dh_ddiff):
	return dpprime_dp.T.dot((diff_generating_mat.T.dot(dh_ddiff)))

def update_delta(delta_t, Q, params, training_data):
	return (1 - params["teleport_prob"]) * Q.dot(delta_t) + params["teleport_prob"] / training_data["num_nodes"]

def backprop_grad_one_source(s, p_warm_start, w, training_data, params):
	
	#First we recompute Q and its gradients
 	(A, A_data) = edge_computation.compute_A(w, training_data["feature_stack"], training_data["edge_ij"], params["edge_strength_fun"], training_data["num_nodes"])
	Q = edge_computation.compute_Q(A, A_data, training_data["edge_ij"], s, training_data["num_nodes"], params)

	p = []
	if params["stationary_p"]: #if bottom layer has stationary p (which means all layers have that p)
		p = page_rank_update.update_p(p_warm_start, Q, params)
	else:
		print("WE'RE TOO LAZY TO IMPLEMENT THAT NOW!!!")
		assert(0)

	dF_dw = numpy.zeros((training_data["num_features"], 1))
	if params["stationary_p"]:
		p_T_dQ_dw = numpy.zeros((training_data["num_features"], training_data["num_nodes"]))
		for k in range(training_data["num_features"]):
			(df_dwk, df_dwk_data) = edge_computation.compute_df_dwk(k, training_data["feature_stack"], A_data, training_data["edge_ij"], params["edge_strength_grad_fun"], training_data["num_nodes"], params)
			Q_grad_k = edge_computation.compute_dQ_dwk(k, df_dwk, df_dwk_data, A, params, training_data["edge_ij"], A_data)
			p_T_dQ_dw[k, :] = Q_grad_k.T.dot(p).T

		(p_prime, sum_p_candidates) = grad_one_source.compute_p_prime(training_data, p)
		diffs = training_data["diff_generating_mat"].dot(p_prime)
		dh_ddiff = params["h_grad_fun"](diffs, params["margin"])
		delta = compute_top_layer_delta(grad_one_source.compute_dpprime_dp(training_data, p), training_data["diff_generating_mat"], dh_ddiff)
		while True:
			dF_dw_last = numpy.copy(dF_dw)
			dF_dw += numpy.dot(p_T_dQ_dw, delta)
			#print(dF_dw - dF_dw_last)
			delta = update_delta(delta, Q, params, training_data)			
			max_diff = numpy.amax(numpy.fabs(dF_dw - dF_dw_last))
			if max_diff < params["backprop_epsilon"]:
				break
	else:
		print("WE'RE TOO LAZY TO IMPLEMENT THAT NOW!!!")
		assert(0)

	return (dF_dw, p, None)
